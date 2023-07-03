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
import argparse
from pathlib import Path

import gradio as gr
import mdtex2html
import torch

_work_dir = str(Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.insert(0, _work_dir)

from src.models import Models
from src.quantization import load_quantized_for_inference

BALANCE_FORMAT = '''<p align="center"><strong>Token Balance: [{}/{}]</strong></p>'''


class Chatbot(gr.Chatbot):

    def __init__(self, value=None, color_map=None, **kwargs):
        super().__init__(value, color_map, **kwargs)

    def postprocess(self, y):
        if y is not None:
            processed_messages = [
                [
                    None if message is None else mdtex2html.convert(message),
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
        past_key_values,
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
        response, balance = model.chat(
            tokenizer,
            query,
            prompt=prompt,
            history=history,
            query_prefix=query_prefix,
            answer_prefix=answer_prefix,
            **generation_params,
        )
    format_query, text = parse_text(query), parse_text(response)
    chatbot.append([format_query, text])
    history.append([format_query, text])

    return chatbot, history, None, BALANCE_FORMAT.format(balance, max_length)


@torch.no_grad()
def stream_chat(
        past_key_values,
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
        query_prefix,
        answer_prefix,
):
    generation_params = {
        'do_sample': do_sample,
        'num_beams': 1,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'top_p': top_p,
        'top_k': top_k,
        'max_length': max_length,
    }

    chatbot.append((query, ""))
    response, balance = None, 'None'
    for response, history, past_key_values, balance in model.stream_chat(
            tokenizer,
            query=query,
            history=history,
            past_key_values=past_key_values,
            return_past_key_values=True,
            **generation_params,
    ):
        chatbot[-1] = (query, response)

        yield chatbot, history, past_key_values, BALANCE_FORMAT.format(balance, max_length)

    chatbot[-1] = (parse_text(query), parse_text(response))  # 只在结束时格式化,减少正则次数

    return chatbot, history, past_key_values, BALANCE_FORMAT.format(balance, max_length)

    # streamer = TextIteratorStreamer(tokenizer, timeout=TIME_OUT, skip_prompt=True, skip_special_tokens=True)
    # input_ids = models.prepare_forward_inputs(
    #     tokenizer,
    #     query,
    #     prompt,
    #     history=copy.deepcopy(history),
    #     query_prefix=query_prefix,
    #     answer_prefix=answer_prefix,
    # )
    # generation_params = {
    #     'inputs': input_ids,
    #
    #     'streamer': streamer,
    #     'do_sample': do_sample,
    #     'num_beams': 1,
    #     'temperature': temperature,
    #     'repetition_penalty': repetition_penalty,
    #     'top_p': top_p,
    #     'top_k': top_k,
    #     'max_length': max_length,
    # }
    #
    # with torch.cuda.amp.autocast(): models.generate(**generation_params)
    #
    # chatbot.append([query, ''])
    # history.append([query, ''])
    # partial_text = ''
    #
    # for new_text in streamer:
    #     partial_text += new_text
    #     text = parse_text(partial_text)
    #     history[-1][1] = text
    #     chatbot[-1][1] = text
    #
    #     yield chatbot, history, None, None


def clean_trigger():
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # history, chatbot, user_input, past_key_values, token balance
    return [], [], '', None, BALANCE_FORMAT.format('None', 'None')


def build_ui(inf_fn):
    with gr.Blocks(theme=gr.themes.Default(), title='Angel') as gr_webui:
        # uid = gr.State(str(uuid4()))

        # 0. title
        gr.HTML("""<h1 align="center">🪽</h1>""")

        # 1. history
        chatbot = Chatbot(height=560)
        history = gr.State([])
        past_key_values = gr.State(None)

        # 2 input
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

        # 3. function button
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    stop_btn = gr.Button("🛑")
                    submit_btn = gr.Button("🚀")
                    clear_btn = gr.Button("🧹")
                    balance = gr.HTML(
                        value=BALANCE_FORMAT.format('None', 'None'),
                        max_lines=1,
                        container=True,
                    )

        # 4. Generation params
        with gr.Row():
            with gr.Accordion('Control generation', open=False):
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
                                value=0.95,
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
                                value=1.,
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
                                value=0.8,
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
                                value=50,
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
                                maximum=32768,
                                value=8192,
                                step=1.0,
                                label='Maximum length',
                                interactive=True,
                                info=(
                                    'The maximum length the generated tokens can have. '
                                    'Corresponds to the length of the input prompt + max_new_tokens. '
                                    'Its effect is overridden by max_new_tokens, if also set. '
                                    'Click `🧹` to restart a new dialogue'),
                            )

        # 5. prompt
        with gr.Accordion('Instruct chatbot', open=False):
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

        # 6. trigger
        chat_fn_inputs = [
            past_key_values,
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
            input_prefix,
            answer_prefix,
        ]
        trigger_params = {
            'fn': inf_fn,
            'inputs': chat_fn_inputs,
            'outputs': [chatbot, history, past_key_values, balance],
            'queue': True,
            'show_progress': True,
        }
        clean_input_params = {
            'fn': lambda: '',
            'outputs': [user_input],
            'queue': True,
        }
        submit_enter_event = user_input.submit(**trigger_params)
        submit_click_event = submit_btn.click(**trigger_params)
        submit_enter_event.then(**clean_input_params)
        submit_click_event.then(**clean_input_params)

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_enter_event, submit_click_event],
        )
        clear_btn.click(
            fn=clean_trigger,
            outputs=[history, chatbot, user_input, past_key_values, balance],
        )

    return gr_webui


if __name__ == '__main__':
    path = 'THUDM/chatglm2-6b'

    # 0. Params
    parser = argparse.ArgumentParser('ChatWithAngel WebUI')
    parser.add_argument('--model', type=str, default=r'chatglm2')
    parser.add_argument('--model_dir', type=str, default=path, help='The LLM dir')
    parser.add_argument('--quantization_type', type=str, default='hf', choices=['hf', 'gpt'])
    parser.add_argument('--bits', type=int, default=16, choices=[4, 8, 16])

    parser.add_argument('--public_share', type=bool, default=False)
    parser.add_argument('--inbrowser', type=bool, default=True)
    parser.add_argument('--port', type=int, default=7860)

    parser.add_argument('--chat_mode', type=str, default='stream_chat', choices=['chat', 'stream_chat'])

    args, _ = parser.parse_known_args()
    if args.chat_mode == 'chat':
        inference_fn = chat
    elif args.chat_mode == 'stream_chat':
        inference_fn = stream_chat
    else:
        raise NotImplementedError(f'The `{args.chat_mode}` is not implemented. Choices: chat, stream_chat')

    # 1. load models and tokenizer
    model_class, tokenizer_class, _ = Models.get(args.model, [None] * 3)
    if model_class is None: raise ValueError(f'model: {args.model} is  not supported')
    tokenizer = tokenizer_class.from_pretrained(args.model_dir)

    # 2. Quantize model
    if args.bits >= 16:
        model = model_class.from_pretrained(args.model_dir).half().cuda()
    elif args.quantization_type == 'hf' and args.bits < 16:
        model = load_quantized_for_inference(path=args.model_dir, model_class=model_class, bits=args.bits)
    elif args.quantization_type == 'gpt':
        model = model_class.from_pretrained(args.model_dir).quantize(args.bits).cuda()
    else:
        raise NotImplementedError(f'The `{args.quantization_type}` is not implemented, you can choice `hf` or `gpt`.')
    model = model.eval()

    # 3. Build ui
    gr_webui = build_ui(inf_fn=inference_fn)

    # 4. Serve
    gr_webui.queue().launch(
        share=args.public_share,
        inbrowser=args.inbrowser,
        server_port=args.port,
    )

# TODO: 1. optimize code of baichuan
# 3. GPTQ for baichuan
# 4. adjust baichuan stream chat & chat
# 5. use input/answer prefix, prompt after tuning model
