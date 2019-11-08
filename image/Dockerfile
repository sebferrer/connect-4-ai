FROM centos:centos7
LABEL   MAINTAINER="sebferrer" \
         IMAGE_NAME="keras-daemon"

ARG RPMS='shadow-utils.x86_64 procps python3-pip'
ARG PIP_PKG='tensorflow==1.13.1 keras==2.2.4 flask==1.0.2 flask-cors==3.0.7 numpy==1.16.1 pandas==0.24.2'

WORKDIR /

RUN mkdir -p /opt/keras-daemon/files &&\
    rpm --rebuilddb &&\
    yum install -y --nogpgcheck ${RPMS} &&\
    python3 -m pip install --upgrade pip &&\
    python3 -m pip install ${PIP_PKG} &&\
    yum -y clean all

COPY server.py /opt/keras-daemon/server/server.py
COPY training_files /opt/keras-daemon/files

RUN chmod -R 755 /opt &&\
    chmod -R 777 /opt/keras-daemon &&\
    chmod -R 777 /tmp &&\
    groupadd -g 5555 tensorflow &&\
    adduser -u 5555 -g 5555 -d /home/tensorflow tensorflow

USER 5555

EXPOSE 5001

ENTRYPOINT ["python3", "-u", "/opt/keras-daemon/server/server.py"]
