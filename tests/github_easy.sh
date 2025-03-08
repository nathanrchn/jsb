#!/bin/bash

python3 -m bench --engine gemini --config tests/configs/gemini.yaml --tasks Github_easy --limit 50
python3 -m bench --engine guidance --config tests/configs/guidance.yaml --tasks Github_easy --limit 50
python3 -m bench --engine llama_cpp --config tests/configs/llama_cpp.yaml --tasks Github_easy --limit 50
python3 -m bench --engine outlines --config tests/configs/outlines.yaml --tasks Github_easy --limit 50
python3 -m bench --engine xgrammar --config tests/configs/xgrammar.yaml --tasks Github_easy --limit 50
