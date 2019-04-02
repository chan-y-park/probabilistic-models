FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN apt-get update
RUN apt-get -y install vim
RUN apt-get -y install sudo

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

USER docker

WORKDIR /home/docker

CMD /bin/bash
