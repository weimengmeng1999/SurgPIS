FROM nvcr.io/nvidia/pytorch:24.03-py3
# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ARG USER_ID
ARG GROUP_ID
ARG USER
ARG DEBIAN_FRONTEND=noninteractive

RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos "" --uid $USER_ID --gid $GROUP_ID $USER

WORKDIR /nfs/home/$USER/SurgPIS
EXPOSE 8888

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

#detectron2 install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip3 install --upgrade -r requirements.txt
RUN pip install "opencv-python-headless<4.3"
RUN pip install "numpy<2"

# RUN echo $(ls -a)#
ADD mask2former .
ENV FORCE_CUDA="1"
RUN cd modeling/pixel_decoder/ops && sh make.sh
