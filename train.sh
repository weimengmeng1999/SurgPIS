python3 train_net.py \
  --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml \
  --num-gpus 2 WSL.TRAIN_WSL False \
  OUTPUT_DIR /nfs/home/mwei/SurgPIS_output/output_inp
#2 gpu training
# python3 train_net.py \
#   --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --num-gpus 2 \
#   --num-machines 1 \
#   WSL.TRAIN_WSL True \
#   WSL.TEACHER_CKPT /nfs/home/mwei/SurgPIS_output/output_inp_3/model_final.pth \
#   OUTPUT_DIR /nfs/home/mwei/SurgPIS_output/output_inp \
# 1 gpu training (remember to change train_step model.modules)
# python3 train_net.py \
#   --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --num-gpus 1 WSL.TRAIN_WSL True WSL.TEACHER_CKPT /nfs/home/mwei/SurgPIS_output/output_inp_3/model_final.pth OUTPUT_DIR /nfs/home/mwei/SurgPIS_output/output_inp_7 WSL.BURNIN_ITER 100