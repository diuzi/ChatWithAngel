#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@author     : jdi146
@contact    : jdi147.com@gmail.com
@datetime   : 2023/6/16 下午9:26
@description: 
"""

from pathlib import Path
from typing import Union

import torch
from transformers import AutoModel
from loguru import logger


def load_quantized_for_inference(path: Union[Path, str], model_class=AutoModel, bits: int = 4, **kwargs):
    """

    Args:
        model_class:
        path:
        bits:
        **kwargs:

    Returns:

    """
    if bits is not None:
        if not isinstance(bits, (float, int)): raise ValueError('The bit width should be an integer, 4 or 8')
        bits = int(bits)
        if bits not in {4, 8}: raise ValueError(f'The bit width is illegal, it should be 4 or 8')

    free_in_gb = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f"{free_in_gb - 2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    logger.info(f'Max memory: {max_memory}')

    major, minor = torch.cuda.get_device_capability()
    logger.info(f'major: {major}, minor: {minor}')
    torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
    logger.info(f'torch_dtype: {torch_dtype}')

    if isinstance(model_class, AutoModel): kwargs['trust_remote_code'] = True

    model = model_class.from_pretrained(
        path,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        # max_memory=max_memory,
        device_map='auto',
        torch_dtype=torch_dtype,
        **kwargs,
    )

    return model
