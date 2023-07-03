#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@author     : jdi146
@contact    : jdi147.com@gmail.com
@datetime   : 2023/6/16 下午7:36
@description: 
"""

from src.models.baichuan.tokenization_baichuan import BaiChuanTokenizer
from src.models.baichuan.modeling_baichuan import BaiChuanChatBot
from src.models.baichuan.configuration_baichuan import BaiChuanConfig

from src.models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from src.models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
from src.models.chatglm2.configuration_chatglm import ChatGLMConfig

Models = {
    'baichuan': [
        BaiChuanChatBot,
        BaiChuanTokenizer,
        BaiChuanConfig,
    ],
    'chatglm2': [
        ChatGLMForConditionalGeneration,
        ChatGLMTokenizer,
        ChatGLMConfig,
    ]
}
