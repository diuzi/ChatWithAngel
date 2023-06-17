#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@author : jdi146
@contact : jdi147.com@gmail.com
@datetime : 2023/6/16 19:04
@description:
"""

import os
import sys
import copy
import argparse
from pathlib import Path

import gradio as gr
import mdtex2html
import torch
from transformers import TextIteratorStreamer

_work_dir = str(Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.insert(0, _work_dir)

from src.model.modeling_baichuan import BaiChuanChatBot
from src.model.tokenization_baichuan import BaiChuanTokenizer
from src.quantization import gpt_quantize, load_quantized_for_inference


class Chatbot(gr.Chatbot):

    def __init__(self, value=None, color_map=None, **kwargs):
        super().__init__(value, color_map, **kwargs)

    def postprocess(self, y):
        if y is not None:
            processed_messages = [
                [
                    None if message is None else mdtex2html.convert((message)),
                    None if response is None else mdtex2html.convert(response),
                ]
                for message, response in y
            ]
        else:
            processed_messages = []

        return processed_messages


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def chat(
        prompt,
        query,
        history,
        chatbot,
        do_sample,
        num_beams,
        temperature,
        repetition_penalty,
        top_p,
        top_k,
        max_length,
        add_prefix4single,
        query_prefix,
        answer_prefix,
):
    generation_params = {
        'do_sample': do_sample,
        'num_beams': num_beams,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'top_p': top_p,
        'top_k': top_k,
        'max_length': max_length,
    }

    with torch.cuda.amp.autocast():
        outputs = model.chat(
            tokenizer,
            query,
            prompt=prompt,
            history=history,
            add_prefix4single=add_prefix4single,
            query_prefix=query_prefix,
            answer_prefix=answer_prefix,
            **generation_params,
        )
    text = parse_text(outputs)
    chatbot.append([query, text])
    history.append([query, text])

    return chatbot, history


@torch.no_grad()
def stream_chat(
        prompt,
        query,
        history,
        chatbot,
        do_sample,
        num_beams,
        temperature,
        repetition_penalty,
        top_p,
        top_k,
        max_length,
        add_prefix4single,
        query_prefix,
        answer_prefix,
):
    streamer = TextIteratorStreamer(tokenizer, timeout=TIME_OUT, skip_prompt=True, skip_special_tokens=True)
    input_ids = model.prepare_inputs(
        tokenizer,
        query,
        prompt,
        history=copy.deepcopy(history),
        add_prefix4single=add_prefix4single,
        query_prefix=query_prefix,
        answer_prefix=answer_prefix,
    )
    generation_params = {
        'inputs': input_ids,

        'streamer': streamer,
        'do_sample': do_sample,
        'num_beams': num_beams,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'top_p': top_p,
        'top_k': top_k,
        'max_length': max_length,
    }

    with torch.cuda.amp.autocast(): model.generate(**generation_params)

    chatbot.append([query, ''])
    history.append([query, ''])
    partial_text = ''

    for new_text in streamer:
        partial_text += new_text
        text = parse_text(partial_text)
        history[-1][1] = text
        chatbot[-1][1] = text

        yield chatbot, history


def clean_trigger():
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return [], [], ''


def build_ui(inf_fn):
    with gr.Blocks(theme=gr.themes.Default(), title='Welcome') as gr_webui:
        # uid = gr.State(str(uuid4()))

        # 0. title
        gr.HTML("""<h1 align="center">🚀BaiChuanChatBot👻</h1>""")

        # 1. prompt
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    usr_prompt = gr.Textbox(
                        value='',
                        placeholder="Input an instruction",
                        label='Prompt',
                        max_lines=36,
                        container=True,
                        show_copy_button=True,
                    )
            with gr.Column():
                with gr.Row():
                    input_prefix = gr.Textbox(
                        value='问：',
                        label='The input prefix',
                        max_lines=1,
                        container=True,
                        show_copy_button=True,
                    )
            with gr.Column():
                with gr.Row():
                    answer_prefix = gr.Textbox(
                        value='答：',
                        label='The answer prefix',
                        max_lines=1,
                        container=True,
                        show_copy_button=True,
                    )

        # 2. history
        chatbot = Chatbot(height=480)

        # 3 input
        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder='Input your question',
                    lines=2,
                    max_lines=512,
                    container=False,
                    show_copy_button=True,
                )

        # 4. function button
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    stop_btn = gr.Button("🛑")
                    submit_btn = gr.Button("🚀")
                    clear_btn = gr.Button("🧹")

        # 5. Generation params
        with gr.Row():
            with gr.Accordion('Generation params:', open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            do_sample = gr.Checkbox(
                                label='Do sample',
                                value=True,
                                info='Whether or not to use sampling ; use greedy decoding otherwise.',
                            )
                    with gr.Column():
                        with gr.Row():
                            add_prefix4single = gr.Checkbox(
                                label='Add prefix in single dialogue',
                                value=False,
                                info=(
                                    'Add prefix token in single dialogue, '
                                    'in mutil dialogue it will be used no matter what.'
                                    r'Used in Single dialogue: `问：xxx\n答：`; Mutil dialogue: `问：xxx\n答：xxx\n问：xxx答：`'
                                )
                            )
                    with gr.Column():
                        with gr.Row():
                            num_beams = gr.Slider(
                                minimum=1,
                                maximum=16,
                                value=1,
                                step=1,
                                label='Num beams',
                                interactive=True,
                                info='Number of beams for beam search. 1 means no beam search',
                            )
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label='Temperature',
                                value=0.5,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                interactive=True,
                                info=(
                                    'The value used to modulate the next token probabilities. '
                                    'Higher values produce more diverse outputs'
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label='Repetition penalty',
                                value=1.3,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info='The parameter for repetition penalty. 1.0 means no penalty',
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p",
                                value=0.7,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    'If set to float < 1, only the smallest set of most probable tokens with probabilities '
                                    'that add up to top_p or higher are kept for generation'
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label='Top-k',
                                value=0,
                                minimum=0.0,
                                maximum=256,
                                step=1,
                                interactive=True,
                                info='Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.',
                            )
                    with gr.Column():
                        with gr.Row():
                            max_length = gr.Slider(
                                minimum=0,
                                maximum=4096,
                                value=1024,
                                step=1.0,
                                label='Maximum length',
                                interactive=True,
                                info=(
                                    'The maximum length the generated tokens can have. '
                                    'Corresponds to the length of the input prompt + max_new_tokens. '
                                    'Its effect is overridden by max_new_tokens, if also set. '
                                    'Click `🧹` to restart a new dialogue'),
                            )

        # 6. trigger
        history = gr.State([])
        chat_fn_inputs = [
            usr_prompt,
            user_input,
            history,
            chatbot,
            do_sample,
            num_beams,
            temperature,
            repetition_penalty,
            top_p,
            top_k,
            max_length,
            add_prefix4single,
            input_prefix,
            answer_prefix,
        ]
        trigger_params = {
            'fn': inf_fn,
            'inputs': chat_fn_inputs,
            'outputs': [chatbot, history],
            'queue': True,
            'show_progress': True,
        }
        start_params = {
            'fn': lambda: '',
            'outputs': [user_input],
            'queue': False,
            'show_progress': True,
        }
        submit_enter_event = user_input.submit(**trigger_params)
        submit_click_event = submit_btn.click(**trigger_params)
        submit_enter_event.then(**start_params)
        submit_click_event.then(**start_params)

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_enter_event, submit_click_event],
            queue=False,
        )
        clear_btn.click(
            fn=clean_trigger,
            outputs=[history, chatbot, user_input],
            queue=False,
        )

    return gr_webui


if __name__ == '__main__':
    # 0. Params
    parser = argparse.ArgumentParser('BaiChuanChatBot WebUI')
    parser.add_argument('--model_dir', type=str, default=r'baichuan-inc/baichuan-7B', help='The LLM dir')
    parser.add_argument('--quantization_type', type=str, default='hf', choices=['hf', 'gpt'])
    parser.add_argument('--bits', type=int, default=None, choices=[4, 8])

    parser.add_argument('--public_share', type=bool, default=False)
    parser.add_argument('--inbrowser', type=bool, default=True)
    parser.add_argument('--port', type=int, default=7860)

    parser.add_argument('--chat_mode', type=str, default='stream_chat', choices=['chat', 'stream_chat'])
    parser.add_argument('--stream_time_out', type=float, default=20.)

    args, _ = parser.parse_known_args()
    TIME_OUT = args.stream_time_out
    if args.chat_mode == 'chat':
        inference_fn = chat
    elif args.chat_mode == 'stream_chat':
        inference_fn = stream_chat
    else:
        raise NotImplementedError(f'The `{args.chat_mode}` is not implemented. Choices: typewriter, chat, stream_chat')

    # 1. load model and tokenizer
    tokenizer = BaiChuanTokenizer.from_pretrained(args.model_dir)
    # print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) # pad_token_id is None
    if args.quantization_type == 'hf':
        model = load_quantized_for_inference(path=args.model_dir, model_class=BaiChuanChatBot, bits=args.bits)
    elif args.quantization_type == 'gpt':
        torch_type_ = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
        model = gpt_quantize(
            BaiChuanChatBot.from_pretrained(args.model_dir, ),  # torch_dtype=torch_type_, device_map='auto'
            weight_bit_width=args.bits,
            empty_init=False,
        ).half()
        if model.device == 'cpu' and torch.cuda.is_available(): model = model.cuda()
    else:
        raise NotImplementedError(f'The `{args.quantization_type}` is not implemented, you can choice `hf` or `gpt`.')

    model = model.eval()

    # 2. build ui
    gr_webui = build_ui(inf_fn=inference_fn)

    # 3. serve
    gr_webui.queue().launch(
        share=args.public_share,
        inbrowser=args.inbrowser,
        server_port=args.port,
    )

# TODO： 1. 流式时间对不上，展示延迟； 2. gptq量化测试； 3. 当前对话可生成token数量; 2023.06.17
