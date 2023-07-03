#!/usr/bin/env bash

inbrowser=true

python src/ui/webui.py \
  --quantization_type hf \
  --inbrowser inbrowser\
  --overwrite_cache \
  --port 7860 \
