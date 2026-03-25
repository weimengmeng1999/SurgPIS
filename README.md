# SurgPIS: Surgical-instrument-level Instances and Part-level Semantics for Weakly-supervised Part-aware Instance Segmentation
SurgPIS redefine the task of surgical instrument segmentation as a unified part-aware instance segmentation problem, and propose SurgPIS, the first model for part-aware instance segmentation of surgical tools.

# Introduction 📑
SurgPIS is capable of predicting both instrument-level instances (i.e. IIS and ISS) and part-aware instances (i.e. PIS and PSS) based on a query- based transformation approach. It leverages weak PIS supervision from disjoint PSS and ISS datasets to learn from partially labelled data in different granularities.
<div align="center">
  <img src="https://github.com/weimengmeng1999/SurgPIS/blob/main/Figures/open.png" width="80%" height="80%"/>
</div><br/>

# Dataset 📊
The dataset is organized as follows for this code:

```text
Datasets/
└── EndoVis2018/ (same for EndoVis2017) 
    ├── train/
    │   ├── images/
    │   │   └── *.png (original image files) 
    │   └── pss_annotations/
    │   │   └── *.png (part-level semantic (PSS) annotation files) 
    │   └── pis_annotations/
    │       └── *.png (part-level instance (PIS) annotation files) 
    ├── test/
    │   ├── images/
    │   │   └── *.png (original image files) 
    │   └── pss_annotations/
    │   │   └── *.png (part-level semantic (PSS) annotation files) 
    │   └── pis_annotations/
    │       └── *.png (part-level instance (PIS) annotation files) 
    │   └── endovis_instance_train.json
    │   └── endovis_instance_test.json
└── SAR-RARP50/ 
    ├── train/
    │   ├── images/
    │   │   └── *.png (original image files) 
    │   └── pss_annotations/
    |   |   └── *.png (part-level semantic (PSS) annotation files) 
    ├── test/
    │   ├── images/
    │   │   └── *.png (original image files) 
    │   └── pss_annotations/
            └── *.png (part-level semantic (PSS) annotation files) 
└── GraSP/ 
    └── test/
        ├── images/
        │   └── *.png (original image files) 
        └── pis_annotations/
            └── *.png (part-level instance (PIS) annotation files) 
```

# How to Run the Code 🛠
## Environment Installation
Our codebase is based on detectron2 and Mask2Former. 
Please follow the docker file provided in [Dockerfile](Dockerfile).

## Model training
Training is split into two steps: 
* **Fully-supervised PIS stage (stage 1)** : Fully supervised training for PIS task on EndoVis2018 dataset.
* **Weakly-supervised stage (stage 2)** : Weakly supervised training for dataset that only has IIS annotations (EndoVis2017) or PSS annotations (SAR-RARP50).

The following section provides examples of scripts to train for different stages.

### Example with a R50 backbone
* Example with ResNet-50 as backbone on for stage 1 on 1 GPU.

    ```
    python3 train_net.py --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml --num-gpus 1 WSL.TRAIN_WSL False OUTPUT_DIR *OUTPUT/TEACHER*
    ```


* Example with ResNet-50 as backbone for stage 2 on 2 GPUs.

    ```
    python3 train_net.py --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml --num-gpus 2 --num-machines 1 WSL.TRAIN_WSL True WSL.TEACHER_CKPT *OUTPUT/TEACHER/model_final.pth* OUTPUT_DIR *OUTPUT/STUDENT*
    ```

### Example with a a DINOv2 pre-trained backbone
* Example with DINOv2 pre-trained ViT-B as backbone on for stage 1 on 1 GPU.

    ```
    python3 train_net.py --config-file configs/endovis/instance-part-segmentation/maskformer2_dinov2_base_bs16_50ep.yaml --num-gpus 1 WSL.TRAIN_WSL False OUTPUT_DIR *OUTPUT/TEACHER*
    ```

* Example with ResNet-50 as backbone for stage 2 on 2 GPUs.

    ```
    python3 train_net.py --config-file configs/endovis/instance-part-segmentation/maskformer2_dinov2_base_bs16_50ep.yaml --num-gpus 2 --num-machines 1 WSL.TRAIN_WSL True WSL.TEACHER_CKPT *OUTPUT/TEACHER/model_final.pth* OUTPUT_DIR *OUTPUT/STUDENT*
    ```
## Model evaluation
Add 'eval_only' to evaluate the corresponding experiments, and define which model that you want to evaluate by changing 'WSL.EVAL_WHO'

* Example with ResNet-50 as backbone on for stage 1 on 1 GPU.

    ```
    python train_net.py --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS /nfs/home/mwei/SurgPIS_output/output_inp_13/model_0027999.pth WSL.TRAIN_WSL False WSL.EVAL_WHO TEACHER
    ```

# Citation 📖
If you find our work useful or relevant to your research, please consider citing:
```
@article{wei2025SurgPIS,
  title={SurgPIS: Surgical-instrument-level Instances and Part-level Semantics for Weakly-supervised Part-aware Instance Segmentation},
  author={Meng Wei, Charlie Budd, Oluwatosin Alabi, Miaojing Shi, and Tom Vercauteren},
  year={2025},
}
```