ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

USER root

RUN apt-get update

USER ${NB_UID}

COPY requirements.txt /tmp/
# Install Python 3 packages
RUN conda install --quiet --yes --file /tmp/requirements.txt
RUN fix-permissions "${CONDA_DIR}" && fix-permissions "/home/${NB_USER}"
RUN conda install gcc --quiet --yes
COPY requirements.dev.txt /tmp/
RUN pip install -r /tmp/requirements.dev.txt
## Install facets which does not have a pip or conda package at the moment
WORKDIR /tmp
RUN git clone https://github.com/PAIR-code/facets.git && \
    jupyter nbextension install facets/facets-dist/ --sys-prefix && \
    rm -rf /tmp/facets && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions "/home/${NB_USER}"

RUN python -c "import nltk; nltk.download('punkt')"

RUN rm -rf /home/${NB_USER}/work/

USER ${NB_UID}

WORKDIR "${HOME}"

