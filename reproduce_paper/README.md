# Reproduce Results in "JsonSchemaBench" Paper


## Setup

Install the packages with the same versions as in the paper. 
```bash
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" pip install llama-cpp-python==0.3.1
```

## Reproduce Results

```bash
bash reproduce_paper/reproduce.sh
```