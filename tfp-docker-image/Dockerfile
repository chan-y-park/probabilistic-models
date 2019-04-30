FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

RUN apt-get update
RUN apt-get -y install vim
RUN apt-get -y install sudo
RUN apt-get -y install git

# Installing Bazel
# https://docs.bazel.build/versions/master/install-ubuntu.html
RUN apt-get -y install openjdk-8-jdk
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt-get update && apt-get -y install bazel

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Installing tf-probability
# run the following if tfp git repo is not cloned already. 
# WORKDIR /home/docker
# git clone https://github.com/tensorflow/probability.git tf-probability
COPY tfp-git-repo /usr/src/tf-probability
WORKDIR /usr/src/tf-probability
RUN git checkout r0.6
RUN bazel build --copt=-O3 --copt=-march=native :pip_pkg
RUN mkdir pkg
RUN ./bazel-bin/pip_pkg pkg 
RUN pip install --upgrade pkg/*.whl

RUN pip install jupyter_contrib_nbextensions 

# Log in as user 'docker'.
USER docker
WORKDIR /home/docker
COPY vimrc .vimrc
CMD /bin/bash
