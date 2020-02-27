# syntax=docker/dockerfile:1.0.0-experimental
FROM continuumio/miniconda3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root
WORKDIR $HOME

COPY . $HOME/
RUN conda env update -f $HOME/environment.yml
RUN pip install -e ".[dev]"
#RUN pip install -e ".[reproduce]"

CMD ["pytest", "--color=yes", "-s", "tests/"]
