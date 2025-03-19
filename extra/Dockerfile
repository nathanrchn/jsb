ARG BASE_IMAGE="ghcr.io/jkminder/dlab-runai-images/base:master"

FROM ${BASE_IMAGE} AS base

COPY requirements.txt /tmp/requirements.txt

RUN /opt/conda/bin/mamba create -y -n default && \
    /opt/conda/bin/mamba clean -y --all

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

FROM ${BASE_IMAGE}
COPY --from=base /opt/conda /opt/conda

COPY extra/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
