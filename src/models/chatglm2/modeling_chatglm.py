""" PyTorch ChatGLM model. """

import copy
import warnings
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import skip_init
from typing import (
    Optional,
    Tuple,
    Union,
    List,
    Callable,
    Dict,
    Any
)

from transformers import StoppingCriteria
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    ModelOutput
)

from .configuration_chatglm import ChatGLMConfig
from .quantization import quantize

logger = logging.get_logger(__name__)

# flags required to enable jit fusion kernels
if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


def _config_to_kwargs(args):
    return {
        "dtype": args.torch_dtype,
    }


def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id: return True

        return False


class PrefixEncoder(torch.nn.Module):

    def __init__(self, config: ChatGLMConfig):

        super().__init__()
        self.prefix_projection = config.prefix_projection

        if self.prefix_projection:
            self.proj = torch.nn.Sequential(
                *[
                    torch.nn.Embedding(config.pre_seq_len, config.hidden_size),
                    torch.nn.Linear(config.hidden_size, config.hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(
                        config.hidden_size,
                        config.num_layers * config.kv_channels * config.multi_query_group_num * 2,
                    ),
                ]
            )
        else:
            self.proj = torch.nn.Embedding(
                config.pre_seq_len,
                config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            )

    def forward(self, prefix: torch.Tensor):
        """

        Args:
            prefix: [batch_size, pre_seq_len]

        Returns: [batch_size, pre_seq_len, num_layers * kv_channels * multi_query_group_num * 2]

        """
        return self.proj(prefix)


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return [chunk.contiguous() for chunk in tensor_list]

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings: int = 32768, base: int = 10000, dtype=None):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.max_seq_len_cached = 0
        self.base = base

        self.register(max_position_embeddings)

    def register(self, max_len):
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).to(dtype=self.dtype) / self.dim))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(max_len, dtype=theta.dtype)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if theta.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.int8,
        }:
            cache = cache.bfloat16() if theta.dtype == torch.bfloat16 else cache.half()

        self.max_seq_len_cached = max_len
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, seq_len: int, position_ids: torch.LongTensor):
        if seq_len and seq_len > self.max_seq_len_cached: self.register(seq_len)

        return self.cache[position_ids]


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)

    return torch.cat((x_out2, x_pass), dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()
        self.hidden_size = config.hidden_size

        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """

        Args:
            input_ids: [seq_len, batch_size, hidden_size]

        Returns:

        """
        # [batch_size, seq_len, hidden_size] => [seq_len, batch_size, hidden_size]
        return self.word_embeddings(input_ids).transpose(0, 1).contiguous()


class CoreAttention(torch.nn.Module):

    def __init__(self, config: ChatGLMConfig):
        super(CoreAttention, self).__init__()

        # Equal hidden_size
        self.hidden_size_per_partition = config.kv_channels * config.num_attention_heads

    def forward(self, query_layer, key_layer, value_layer, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """

        Args:
            query_layer: [tgt_len, batch_size, num_heads, head_head_size]
            key_layer: [src_len, batch_size, num_heads, head_head_size]
            value_layer: [src_len, batch_size, num_heads, head_head_size]
            attention_mask: [batch_size, 1, seq_len, src_len]

        Returns: [tgt_len, batch_size, hidden_size]

        """
        # [seq_len, batch_size, num_heads, head_head_size] => [batch_size, num_heads, seq_len, head_head_size]
        query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

        # [batch_size, num_heads, tgt_len, head_hidden_size]
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask
        )

        # [tgt_len, batch_size, num_heads, head_hidden_size]
        context_layer = context_layer.permute(2, 0, 1, 3)

        # [tgt_len, batch_size, num_heads * head_hidden_size]   <=> [tgt_len, batch_size, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(SelfAttention, self).__init__()

        # Per attention head and per partition values.
        self.projection_size = config.kv_channels * config.num_attention_heads
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.num_multi_query_groups_per_partition = config.multi_query_group_num

        # MQA
        self.qkv_hidden_size = self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
        self.query_key_value = nn.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            device=device,
            **_config_to_kwargs(config),
        )
        self.core_attention = CoreAttention(config)

        # Output.
        self.dense = nn.Linear(
            self.projection_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states, attention_mask: torch.BoolTensor, rotary_pos_emb, kv_cache=None, use_cache=True):
        """

        Args:
            hidden_states: [tgt_len, batch_size, hidden_size]
            attention_mask:  [batch_size, 1, tgt_len, src_len]
            rotary_pos_emb: [tgt_len, 1, head_hidden_size // 2 // 2, 2]
            kv_cache:   Tuple:
                        [
                            [src_len, batch_size, num_heads, head_hidden_size],
                            [src_len, batch_size, num_heads, head_hidden_size]
                        ]
            use_cache:

        Returns:

        """
        # [tgt_len, batch_size, hidden_size] =>
        # [tgt_len, batch_size, (q_num_heads * head_hidden_size)
        # + (k_num_heads * head_hidden_size)
        # + (v_num_heads * head_hidden_size)]

        # 0. QKV-dense
        # k_num_heads/v_num_heads: 2
        # [tgt_len, batch_size, 4608(4096 + 128 * 2 * 2)]
        mixed_x_layer = self.query_key_value(hidden_states)

        # 1. Chunk heads for q, k, v
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        # [tgt_len, batch_size, q_num_heads, head_hidden_size]
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        # [tgt_len, batch_size, k_num_heads, head_hidden_size]
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        # [tgt_len, batch_size, v_num_heads, head_hidden_size]
        value_layer = value_layer.view(
            value_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )

        # 2. Apply relative positional encoding (rotary embedding)
        # Note: Before concat past_key_values do this
        if rotary_pos_emb is not None:
            # [tgt_len, batch_size, v_num_heads, head_hidden_size]
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            # [src_len, batch_size, k_num_heads, head_hidden_size]
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # 3. Adjust key and value for inference
        if kv_cache is not None:
            # key_layer/value_layer: [past_key_value_len + input_seq_len, batch_size, kv_num_heads, head_hidden_size]
            key_layer, value_layer = torch.cat((kv_cache[0], key_layer), dim=0), torch.cat((kv_cache[-1], value_layer), dim=0)
        kv_cache = (key_layer, value_layer) if use_cache else None

        # 4. Repeat the number of k/v heads(MQA)
        repeat_cnt = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
        new_shape = value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        key_layer, value_layer = (
            #  [src_len, batch_size, kv_num_heads, head_hidden_size] => [src_len, batch_size, kv_num_heads, 1, head_hidden_size]
            #  [src_len, batch_size, kv_num_heads, 1, head_hidden_size] => [src_len, batch_size, q_num_heads, head_hidden_size]
            item.unsqueeze(-2).expand(-1, -1, -1, repeat_cnt, -1).reshape(new_shape)
            for item in (key_layer, value_layer)
        )

        # 5. Self attention: [tgt_len, batch_size, hidden_size]
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # 6. Dense: [tgt_len, batch_size, hidden_size] -> [tgt_len, batch_size, hidden_size]
        output = self.dense(context_layer)

        return output, kv_cache


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=config.add_bias_linear,
            device=device,
            **_config_to_kwargs(config)
        )

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        """

        Args:
            hidden_states: [tqt_len, batch_size, hidden_size]

        Returns: [tqt_len, batch_size, hidden_size]

        """
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class GLMBlock(torch.nn.Module):

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMBlock, self).__init__()

        # Layernorm on the input data.
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=config.torch_dtype
        )

        # Self attention.
        self.self_attention = SelfAttention(config, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=config.torch_dtype
        )

        # MLP
        self.mlp = MLP(config, device=device)

    def forward(self, hidden_states, attention_mask: torch.BoolTensor, rotary_pos_emb, kv_cache=None, use_cache=True):
        """

        Args:
            hidden_states: [tgt_len, batch_size, hidden_size]
            attention_mask:  [batch_size, 1, tgt_len, src_len]
            rotary_pos_emb: [tgt_len, 1, head_hidden_size // 2 // 2, 2]
            kv_cache:
                        Tuple[
                                   [src_len, batch_size, num_heads, head_hidden_size],
                                   [src_len, batch_size, num_heads, head_hidden_size]
                            ]
            use_cache:

        Returns: Tuple[
                hidden_states: [tqt_len, batch_size, hidden_size],
                kv_cache:   Tuple[
                                   [src_len, batch_size, num_heads, head_hidden_size],
                                   [src_len, batch_size, num_heads, head_hidden_size]
                                ]
        """
        # 0. Embedding token
        # ...

        # 1. Layer norm at the beginning of the transformer layer.
        # [tgt_len, batch_size, hidden_size]
        layernorm_output = self.input_layernorm(hidden_states)

        # 2. Self attention.
        # attention_output: [tgt_len, batch_size, hidden_size]
        # kv_cache: Tuple[
        #     [src_len, batch_size, num_heads, head_hidden_size],
        #     [src_len, batch_size, num_heads, head_hidden_size]
        # ]
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # 3. Residual connection & Norm
        residual = hidden_states + attention_output
        layernorm_output = self.post_attention_layernorm(residual)

        # 4. MLP, [tqt_len, batch_size, hidden_size]
        mlp_output = self.mlp(layernorm_output)

        # 5. Residual connection.
        output = residual + mlp_output

        return output, kv_cache


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):

        super(GLMTransformer, self).__init__()
        self.num_layers = config.num_layers
        self.gradient_checkpointing = False

        self.layers = torch.nn.ModuleList(
            [
                GLMBlock(config, device=device)
                for _ in range(self.num_layers)
            ]
        )

        # Final layer norm before output.
        self.final_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=config.torch_dtype
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.BoolTensor,
            rotary_pos_emb: torch.Tensor,
            kv_caches: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        """

        Args:
            hidden_states: [tgt_len, batch_size, hidden_size]
            attention_mask:  [batch_size, 1, tgt_len, src_len]
            rotary_pos_emb: [tgt_len, 1, head_hidden_size // 2 // 2, 2]
            kv_caches: Tuple[Tuple[torch.Tensor, torch.Tensor]], a tuple of length num_layers
                for tuple, each item is k/v, shape:
                     [past_len, batch_size, kv_num_heads, head_hidden_size]
                     or [pre_seq_len, batch_size, kv_num_heads, head_hidden_size]
            use_cache:
            output_hidden_states:

        Returns:
            Tuple[
                        last_hidden_states: [tgt_len, batch_size, hidden_size],
                        kv_caches: Tuple of Tuple, length is layers-num
                            inner Tuple[
                                   [src_len, batch_size, num_heads, head_hidden_size],
                                   [src_len, batch_size, num_heads, head_hidden_size]
                            ]
                        all_hidden_states: Tuple of [tgt_len, batch_size, hidden_size], length is layers-num
                        None
                ]

        """
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        presents = () if use_cache else None
        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        if not kv_caches: kv_caches = (None for _ in range(self.num_layers))

        for layer, kv_ch in zip(self.layers, kv_caches):
            if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_ch,
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_ch,
                    use_cache=use_cache
                )

            hidden_states, kv_cache = layer_ret
            if use_cache: presents = presents + (kv_cache,)

        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    @staticmethod
    def get_position_ids(input_ids: torch.Tensor, device: Union[torch.device, str]) -> torch.LongTensor:
        """

        Args:
            input_ids:
            device:

        Returns: [batch_size, seq_len]

        """
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

        return position_ids

    @staticmethod
    def _set_gradient_checkpointing(module, value=False):
        if isinstance(module, GLMTransformer): module.gradient_checkpointing = value

    @staticmethod
    def prepare_masks(
            input_shape: Union[torch.Size, Tuple],
            device: Union[torch.device, str],
            past_len: int = 0,
            padding_mask: Optional[torch.Tensor] = None,
            prefix_encoder_prompt_len: int = 0,
    ) -> torch.BoolTensor:
        """Prepare `attn_mask` for <torch.nn.functional.scaled_dot_product_attention>

                # Efficient implementation equivalent to the following:
                attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
                attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
                attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
                attn_weight = torch.dropout(attn_weight, dropout_p)
                return attn_weight @ V

        Args:
            input_shape:
            device:
            past_len:
            padding_mask: The strategy of padding must be 'left'
                for inference: [batch_size, past_len + seq_len]
                for inference: [batch_size, past_len + seq_len]
                for inference: [batch_size, past_len + seq_len]
            prefix_encoder_prompt_len:

        Examples:

            For train:
                    input:
                        input_shape: (2, 5)
                        past_len: 0
                        padding_mask:
                            [
                                [0, 0, 1, 1, 1],
                                [1, 1, 1, 1, 1]
                            ]

                    Output:
                        [[[[False, False, False, False, False],
                          [False, False, False, False, False],
                          [False, False,  True, False, False],
                          [False, False,  True,  True, False],
                          [False, False,  True,  True,  True]]],


                        [[[ True, False, False, False, False],
                          [ True,  True, False, False, False],
                          [ True,  True,  True, False, False],
                          [ True,  True,  True,  True, False],
                          [ True,  True,  True,  True,  True]]]]

            For batch inference:
                    input:
                        input_shape: (2, 5)
                        past_len: 2
                        padding_mask:
                            # past_padding_mask   new_padding_mask
                            [[1, 1,               0, 0, 1, 1, 1],
                            [0, 1,                1, 1, 1, 1, 1]]

                    Output:
                        [[[[ True,  True, False, False, False, False, False],
                          [ True,  True, False, False, False, False, False],
                          [ True,  True, False, False,  True, False, False],
                          [ True,  True, False, False,  True,  True, False],
                          [ True,  True, False, False,  True,  True,  True]]],

                        [[[False,  True,  True, False, False, False, False],
                          [False,  True,  True,  True, False, False, False],
                          [False,  True,  True,  True,  True, False, False],
                          [False,  True,  True,  True,  True,  True, False],
                          [False,  True,  True,  True,  True,  True,  True]]]]
        Returns:

        """
        batch_size, tgt_len = input_shape
        total_len = prefix_encoder_prompt_len + past_len + tgt_len

        # The input_ids's causal mask
        causal_mask = torch.ones(batch_size, tgt_len, tgt_len, device=device).tril_()
        if past_len + prefix_encoder_prompt_len:
            causal_mask = torch.cat(
                [
                    torch.ones(batch_size, tgt_len, past_len + prefix_encoder_prompt_len, device=device),
                    causal_mask
                ],
                dim=-1
            )

        # past padding mask + current padding mask    or    current padding mask
        if padding_mask is not None:
            if padding_mask.size(0) != batch_size:
                raise ValueError('Attention mask should have the same size as the batch size.')

            if prefix_encoder_prompt_len:
                padding_mask = torch.cat(
                    [
                        torch.ones(batch_size, tgt_len, prefix_encoder_prompt_len, device=device),
                        padding_mask
                    ],
                    dim=-1
                )

            if padding_mask.size(1) != total_len:
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, total_len)}, "
                    f"but is {padding_mask.size()}"
                )

            causal_mask = causal_mask * padding_mask.unsqueeze(1)

        # [batch_size, seq_len, src_len] => [batch_size, 1, seq_len, src_len]
        return (causal_mask > 0.5).bool()[:, None, ...]


class ChatGLMModel(ChatGLMPreTrainedModel):

    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)

        init_method = skip_init if empty_init else default_init
        init_kwargs = {'device': device} if device is not None else {}

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Token embedding
        self.embedding = init_method(Embedding, config, **init_kwargs)

        # Rotary positional embeddings
        # TODO: 10000 is Configurable
        self.seq_length = config.seq_length
        self.rotary_pos_emb = RotaryEmbedding(
            max_position_embeddings=config.seq_length,
            base=10000,
            dim=(config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels) // 2,
            dtype=config.torch_dtype,
        )

        # Decoder
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)

        # lm head
        self.output_layer = init_method(
            nn.Linear,
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            **init_kwargs
        )

        # Prefix decoder
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None and self.pre_seq_len > 0:
            # Freeze the llm params while use p-tuning, only prefix-decoder is learned.
            for param in self.parameters(): param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            # TODO: prefix-decoder dropout is Configurable
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def _get_Ptuning_prompt(
            self,
            batch_size: int,
            device: Union[torch.device, str],
            dtype: torch.dtype = torch.half,
    ) -> torch.Tensor:
        """

        Args:
            batch_size:
            device:
            dtype:

        Returns:
                [num_layers, pre_seq_len, batch_size, multi_query_group_num(kv_num_heads), kv_channels(head_hidden_size)]

        """
        """
        batch_size: 2, pre_seq_len: 5
            [0, 1, 2, 3, 4] 
                => 
                    [[0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4]]
        """
        # [pre_seq_len] => [1, pre_seq_len] => [batch_size, pre_seq_len]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)

        # [batch_size, num_layers * kv_channels * multi_query_group_num * 2]
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        past_key_values = self.dropout(past_key_values)

        # [num_layers * 2, pre_seq_len, batch_size, multi_query_group_num(kv_num_heads), kv_channels(head_hidden_size)] =>
        # Tuple: [num_layers, pre_seq_len, batch_size, multi_query_group_num(kv_num_heads), kv_channels(head_hidden_size)],
        # [num_layers, pre_seq_len, batch_size, multi_query_group_num(kv_num_heads), kv_channels(head_hidden_size)]
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)

        return past_key_values

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """

        Args:
            input_ids: [batch_size * seq_len]
            position_ids: [batch_size * seq_len]
            attention_mask: [batch_size * seq_len]
            past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]], a tuple of length num_layers
                for tuple, each item is k/v, shape:
                     [past_len, batch_size, kv_num_heads, head_hidden_size]
                     or [pre_seq_len, batch_size, kv_num_heads, head_hidden_size]
            inputs_embeds: [batch_size * seq_len * hidden_size]
            use_cache:
            output_hidden_states:
            return_dict:

        Returns:

        """
        # 0. Prepare params for return
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embedding(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # 2.  Rotary positional embeddings
        past_key_values_length = past_key_values[0][0].shape[0] if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_ids = position_ids[None, ...]  # [1, seq_len]
        elif position_ids.dim() == 1:
            position_ids = position_ids[None, ...].long()  # [1, seq_len]
        elif position_ids.dim() == 2:
            # [batch_size, seq_len]
            if position_ids.size(1) != seq_length:
                raise ValueError("The position_ids dimension is 2, it should have the same size as input_ids or inputs_embeds.")
        else:
            raise ValueError("Position first dimension must be one of [None, 1, or 2]")

        # [1, seq_len, head_hidden_size // 2 // 2, 2]    # last dim: [cos, sin]
        rotary_pos_emb = self.rotary_pos_emb(past_key_values_length + seq_length, position_ids)
        # [1, seq_len, head_hidden_size // 2 // 2, 2] => [seq_len, 1, head_hidden_size // 2 // 2, 2]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # 3.  Invert attention_mask, add padding mask and expand it: [batch_size, 1, seq_len, src_len]
        if self.pre_seq_len and past_key_values is not None:
            # In the inference, use cache, use `p-tuning v2`, the `past_key_values` contain `prefix-prompt-embedding`
            past_len = past_key_values[0][0].shape[0] - self.pre_seq_len
        elif past_key_values is not None:
            # In the inference, use cache, not use `p-tuning v2`
            past_len = past_key_values[0][0].shape[0]
        else:
            # training or Inference(not use cache)
            past_len = 0
        full_attention_mask: torch.BoolTensor = self.prepare_masks(
            input_shape=(inputs_embeds.size(1), inputs_embeds.size(0)),  # [batch_size, seq_len]
            device=inputs_embeds.device,
            past_len=past_len,
            padding_mask=attention_mask,
            prefix_encoder_prompt_len=self.pre_seq_len if self.pre_seq_len else 0,
        )

        # 4. Continuous prompt embeddings for every layers
        # past_key_values is None <=> first forward for inference / training
        # pre_seq_len is not None and pre_seq_len > 0
        if self.pre_seq_len and past_key_values is None:
            past_key_values = self._get_Ptuning_prompt(
                batch_size=batch_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )

        if past_key_values is None and self.pre_seq_len is not None and self.pre_seq_len > 0:
            past_key_values = self._get_Ptuning_prompt(
                batch_size=batch_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )

        # 5. Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        # 6. Return results
        if not return_dict:
            return tuple(
                v
                for v in (hidden_states, presents, all_hidden_states, all_self_attentions)
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def quantize(self, weight_bit_width: int):
        self.encoder = quantize(self.encoder, weight_bit_width)
        return self


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):

    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        self.config = config
        self.quantized = False

        if self.config.quantization_bit: self.quantize(self.config.quantization_bit, empty_init=True)

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs=outputs,
            standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ],
                dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat([position_ids, new_position_id], dim=-1)

        model_kwargs["is_first_forward"] = False

        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> Dict:
        # only last token for input_ids if past is not None
        if position_ids is None: position_ids = self.get_position_ids(input_ids, device=input_ids.device)

        if not is_first_forward:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True
        }

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
            ignore_index: int = -100,
            **kwargs,
    ):
        """

        Args:
            input_ids: [batch_size * seq_len]
            position_ids: [batch_size * seq_len]
            attention_mask: [batch_size * seq_len]
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], a list of length num_layers
                for tuple, each item is k/v, shape:
                     [past_len, batch_size, kv_num_heads, head_hidden_size]
                     or pre_seq_len, batch_size, kv_num_heads, head_hidden_size]
            inputs_embeds: [batch_size * seq_len * hidden_size]
            labels: [batch_size, seq_len]
            use_cache:
            output_hidden_states:
            return_dict:
            return_last_logit:
            ignore_index:
            kwargs: unexpected keyword argument 'output_attentions', as so on

        Returns:

        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if return_last_logit: hidden_states = hidden_states[-1:]

        # [sql_len, batch_size, hidden_size]
        lm_logits = self.transformer.output_layer(hidden_states)
        # [batch_size, sql_len, hidden_size]
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits, shift_labels = lm_logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                input=shift_logits.view(-1, shift_logits.size(-1)),
                target=shift_labels.view(-1),
                ignore_index=ignore_index,
            )
            lm_logits, loss = lm_logits.to(hidden_states.dtype), loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        if history:
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = input_ids[1:]  # remove 30910， # '_'
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
            """
            # [36474, 54591, 55674]
            input_ids = tokenizer.encode('你好啊', add_special_tokens=False)

            # ['▁你', '好', '啊']
            decoded_ = tokenizer.convert_ids_to_tokens(input_ids))

             # [54591, 55674]
            input_ids = input_ids[1:]  # remove 30910

            # '好啊'
            decoded_ = tokenizer.decode(input_ids,  skip_special_tokens=True)

            # ['好', '啊']
            decoded_ = tokenizer.convert_ids_to_tokens(input_ids))

            # {'input_ids': tensor([[54591, 55674]]), 'attention_mask': tensor([[1, 1]]), 'position_ids': tensor([[0, 1]])}
            
            So, we can see if the first token is not '\n', shift will remove first token
            """
        else:
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")

        return inputs.to(self.device)

    def prepare_forward_inputs(
            self,
            tokenizer,
            query: str,
            prompt: str = '',
            history: Optional[List[List[str]]] = None,
            query_prefix: str = '',
            answer_prefix: str = '',
    ) -> Dict[str, torch.Tensor]:
        """

        Args:
            tokenizer:
            query:
            prompt:
            history:
            query_prefix:
            answer_prefix:

        Returns:

        """
        prompt = prompt.lstrip()

        input_lst = [
            ''.join(
                (
                    query_prefix,
                    old_query,
                    '\n',
                    answer_prefix,
                    old_response,
                    '\n'
                )
            )
            for old_query, old_response in history
        ] if history else []

        cur_input = ''.join((query_prefix, query, '\n', answer_prefix)) if len(history) > 0 else query
        input_lst.append(cur_input)
        input_lst = ([prompt, '\n'] + input_lst) if prompt and prompt[-1] != '\n' else ([prompt] + input_lst)
        raw_input = ''.join(input_lst)

        # Note: add_special_tokens=True is must, either or tokenizer will add some '' for ChatGLMTokenizer
        encode_ = tokenizer([raw_input], add_special_tokens=False, return_tensors='pt').to(self.device)

        return encode_

    def _parse_generation_params(
            self,
            tokenizer,
            *,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            generation_config: Optional[GenerationConfig] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            **kwargs
    ):
        """

        Args:
            tokenizer:
            input_ids:
            logits_processor:
            prefix_allowed_tokens_fn:
            generation_config:
            stopping_criteria:
            **kwargs:

        Returns:

        """
        input_ids_seq_length = input_ids.shape[-1]

        # 0. Check params
        if generation_config is None: generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(InvalidScoreLogitsProcessor())
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
            stop_token_ids = []
            if tokenizer.pad_token_id is not None: stop_token_ids.append(tokenizer.pad_token_id)
            if tokenizer.eos_token_id is not None: stop_token_ids.append(tokenizer.eos_token_id)
            if stop_token_ids: stopping_criteria.append(StopOnTokens(stop_token_ids))

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria
        )
        logits_warper: LogitsProcessorList = self._get_logits_warper(generation_config)

        return {
            'input_ids': input_ids,
            'model_kwargs': model_kwargs,
            'generation_config': generation_config,
            'logits_processor': logits_processor,
            'stopping_criteria': stopping_criteria,
            'logits_warper': logits_warper,
            **kwargs,
        }

    @torch.no_grad()
    def chat(
            self,
            tokenizer,
            query: str,
            prompt: str = '',
            query_prefix: str = '问：\n',
            answer_prefix: str = '答：\n',
            history: Optional[List[List[str]]] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            **kwargs
    ) -> Tuple[str, int]:
        balance = kwargs.get('max_length', self.max_sequence_length)

        if history is None: history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(InvalidScoreLogitsProcessor())
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
            stop_token_ids = []
            if tokenizer.pad_token_id is not None: stop_token_ids.append(tokenizer.pad_token_id)
            if tokenizer.eos_token_id is not None: stop_token_ids.append(tokenizer.eos_token_id)
            if stop_token_ids: stopping_criteria.append(StopOnTokens(stop_token_ids))

        # 1. Prepare inputs
        inputs = self.prepare_forward_inputs(
            tokenizer,
            query,
            prompt,
            history,
            query_prefix=query_prefix,
            answer_prefix=answer_prefix,
        )

        # 2. Forward
        outputs = self.generate(logits_processor=logits_processor, stopping_criteria=stopping_criteria, **inputs, **kwargs)
        outputs = outputs.tolist()[0]
        response = tokenizer.decode(outputs[len(inputs["input_ids"][0]):], skip_special_tokens=True)

        return response, balance - len(outputs)

    @torch.no_grad()
    def stream_chat(
            self,
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = None,
            past_key_values=None,
            return_past_key_values=False,
            max_length: int = 8192,
            do_sample=True,
            top_p=0.8,
            temperature=0.8,
            **kwargs
    ):
        if history is None: history = []
        kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,

            'past_key_values': past_key_values,
            'return_past_key_values': return_past_key_values,

            **kwargs,
        }

        # 0. Prepare inputs
        if past_key_values is None and not return_past_key_values:
            inputs = self.build_inputs(tokenizer, query, history=history)
        else:
            inputs = self.build_stream_inputs(tokenizer, query, history=history)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0]
            if self.transformer.pre_seq_len: past_length -= self.transformer.pre_seq_len
            inputs.position_ids += past_length
            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs['attention_mask'] = attention_mask
        else:
            past_length = 0

        # 1. Prepare gen_kwargs
        params = self._parse_generation_params(tokenizer, **inputs, **kwargs)

        # 2. forward and decode
        cur_len = len(inputs["input_ids"][0])
        for outputs in self.stream_generate(**params):
            if return_past_key_values: outputs, past_key_values = outputs

            outputs = outputs.tolist()[0]
            response = tokenizer.decode(outputs[cur_len:], skip_special_tokens=True).strip()
            if response and response[-1] != "�":
                new_history = history + [(query, response)]
                if return_past_key_values:
                    yield response, new_history, past_key_values, max_length - past_length - len(outputs)
                else:
                    yield response, new_history, max_length - past_length - len(outputs)

    @torch.no_grad()
    def stream_generate(
            self,
            input_ids: torch.LongTensor,
            model_kwargs: Dict[str, Any],
            logits_warper: LogitsProcessorList,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            **kwargs,
    ):
        return_past_key_values = kwargs.get('return_past_key_values', False)

        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            scores = None

            # [batch_size, seq_len, vocab_size]
            outputs = self.forward(**model_inputs, return_dict=True)
            # [batch_size, vocab_size]
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            # [batch_size, vocab_size] -> [batch_size]
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens[:, None].long()], dim=-1)

            # stop when each sentence is finished, or if we exceed the maximum length
            if stopping_criteria(input_ids, scores): break

            # update generated ids, model inputs, and length for next step
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            yield (input_ids, outputs.past_key_values) if return_past_key_values else input_ids

    def quantize(self, bits: int, empty_init=False, device=None):
        if bits is not None:
            if not isinstance(bits, (float, int)): raise ValueError('The bit width should be an integer, 4 or 8')
            bits = int(bits)
            if bits not in {4, 8}: raise ValueError(f'The bit width is illegal, it should be 4 or 8')

        if self.quantized:
            logger.warning("Already quantized.")
        else:
            self.config.quantization_bit = bits
            self.transformer.encoder = quantize(
                self.transformer.encoder,
                bits,
                empty_init=empty_init,
                device=device,
            )
            self.quantized = True

        return self
