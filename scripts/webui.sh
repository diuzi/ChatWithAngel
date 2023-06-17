#!/usr/bin/env bash

inbrowser=true

python src/ui/webui.py \
  --model_dir baichuan-inc/baichuan-7B \
  --quantization_type hf \
  --bits 4 \
  --inbrowser inbrowser\
  --overwrite_cache \
  --port 7860 \
  --stream_time_out 20
