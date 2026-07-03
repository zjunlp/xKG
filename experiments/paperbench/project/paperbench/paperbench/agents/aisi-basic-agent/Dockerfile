FROM pb-env
# from pb-env, we inherit the env vars:
# constants: WORKSPACE_BASE, SUBMISSION_DIR, LOGS_DIR, CODE_DIR, AGENT_DIR
# overridable: CONDA_ENV_NAME, PYTHON_VERSION, REQUIREMENTS

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${REQUIREMENTS}

WORKDIR ${AGENT_DIR}

RUN python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip \
    && pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN conda run -n ${CONDA_ENV_NAME} pip install -r requirements.txt && \
    conda run -n ${CONDA_ENV_NAME} pip install inspect-tool-support && \
    conda run -n ${CONDA_ENV_NAME} inspect-tool-support post-install && \
    conda clean -afy

RUN mkdir -p /opt/

# Install chz from GitHub
# RUN conda run -n ${CONDA_ENV_NAME} pip install "git+https://github.com/openai/chz.git@97cc0dfb5934a4b99c3a96bdcadcfdbe14812fe8#egg=chz"
COPY ./chz /tmp/chz
RUN conda run -n ${CONDA_ENV_NAME} pip install /tmp/chz && \
    rm -rf /tmp/chz  

# Make conda agent packages visible to system Python (needed for Jupyter kernel)
RUN CONDA_SITE=$(/opt/conda/envs/${CONDA_ENV_NAME}/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    SYSTEM_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    echo "$CONDA_SITE" > "$SYSTEM_SITE/conda_agent.pth"

# copy remaining agent files into ${AGENT_DIR}
COPY . ${AGENT_DIR}
