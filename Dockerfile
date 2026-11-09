FROM nvcr.io/nvidia/pytorch:24.03-py3
WORKDIR /workspace/SurgPIS
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' && pip install 'git+https://github.com/cocodataset/panopticapi.git' && pip install -r requirements.txt
COPY . .
ENV FORCE_CUDA=1
RUN cd mask2former/modeling/pixel_decoder/ops && bash make.sh
