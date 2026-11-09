# Installation

The code requires PyTorch 2.x, torchvision, Detectron2, SciPy, pycocotools and the Mask2Former CUDA extension. The server snapshot used NVIDIA PyTorch 24.03; the portable Dockerfile retains that base. A new Docker build has not been validated as part of this cleanup.

```bash
docker build -t surgpis .
docker run --gpus all --shm-size=16g --rm -it -v /path/to/datasets:/workspace/SurgPIS/datasets -v /path/to/output:/workspace/SurgPIS/output surgpis bash
```

For an existing compatible PyTorch/CUDA environment, from the repository root:

```bash
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install 'git+https://github.com/cocodataset/panopticapi.git'
python -m pip install -r requirements.txt
(cd mask2former/modeling/pixel_decoder/ops && bash make.sh)
python train_net.py --help
```

The CUDA extension must be built against the installed PyTorch/CUDA versions. Dependencies fetched from Git are not pinned; record their commits and package versions for a reproducible training run. Use the ResNet stage configs for the documented default; other backbone configs may need separate pretrained weights.
