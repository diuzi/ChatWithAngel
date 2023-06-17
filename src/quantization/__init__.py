#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@author     : jdi146
@contact    : jdi147.com@gmail.com
@datetime   : 2023/6/16 下午8:48
@description: 
"""

from src.quantization.hf_quantization import load_quantized_for_inference

try:
    from src.quantization.gptq import gpt_quantize
except Exception:
    ...
