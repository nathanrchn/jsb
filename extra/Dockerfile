ARG BASE_IMAGE="ghcr.io/jkminder/dlab-runai-images/base:master"

FROM ${BASE_IMAGE}

# install conda environment
COPY extra/env.yaml /tmp/env.yml
COPY requirements.txt /tmp/requirements.txt

RUN /opt/conda/bin/mamba env create -f /tmp/env.yml -n default && \
    /opt/conda/bin/mamba clean -y --all
RUN rm /tmp/env.yml

# install requirements
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate default && \
    mamba clean -y --all && \
    pip install -r /tmp/requirements.txt --no-cache-dir && \
    pip install openai google-generativeai && \
    CMAKE_ARGS="-DGGML_CUDA=on" pip install git+https://github.com/abetlen/llama-cpp-python.git && \
    pip install git+https://github.com/dottxt-ai/outlines.git git+https://github.com/mlc-ai/xgrammar.git && \
    rm /tmp/requirements.txt && \
    chmod -R 777 /opt/conda/envs/default

# install ngrok
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
	| sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
	&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
	| sudo tee /etc/apt/sources.list.d/ngrok.list \
	&& sudo apt update \
	&& sudo apt install ngrok

RUN git clone https://github.com/nathanrchn/jsb.git && cd jsb
