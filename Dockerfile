FROM ubuntu:14.04

MAINTAINER UEI Corporation

ENV TF_BINARY_URL https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
ENV APPROOT /cslaier
ENV CSLAIER_CONFIG $APPROOT/docker_config/cslaier.cfg

RUN apt-get -y update && \
    apt-get -y install \
        python \
        python-dev \
        python-pip \
        python-opencv \
        python-matplotlib \
        sqlite3 \
        libhdf5-dev \
        nkf \
        python-scipy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade $TF_BINARY_URL && \
    mkdir -p $APPROOT

EXPOSE 8080
WORKDIR $APPROOT
COPY ./ $APPROOT
RUN pip install -r requirements.txt && \
    sh setup.sh

# http://stackoverflow.com/questions/31768441/how-to-persist-ln-in-docker-with-ubuntu
CMD sh -c 'ln -s /dev/null /dev/raw1394'; sh run.sh
