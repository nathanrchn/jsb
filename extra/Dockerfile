ARG BASE_IMAGE="ghcr.io/jkminder/dlab-runai-images/pytorch:master"

FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install openai google-generativeai \
    git+https://github.com/guidance-ai/guidance.git@514a5eb16b9d29ad824d9357732ba66e5e767642 \
    && CMAKE_ARGS="-DGGML_CUDA=on" pip install git+https://github.com/abetlen/llama-cpp-python.git \
    && pip install git+https://github.com/dottxt-ai/outlines.git git+https://github.com/mlc-ai/xgrammar.git

RUN git clone https://github.com/nathanrchn/jsb.git $HOME && cd $HOME/jsb
