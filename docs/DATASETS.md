# Dataset files and labels

Defaults expect:

```text
datasets/
  EndoVis2018/
    endovis_instance_train.json
    endovis_instance_val.json
    train/images/*.png
    train/part_annotations/*.png
    val/images/*.png
    val/part_annotations/*.png
  EndoVis2017/
    endovis_instance_train.json
    endovis_instance_val.json
    train/images/*.png
    val/images/*.png
    val/part_annotations/*.png  # only needed for PIS/PartPQ evaluation
  SAR-RARP50/
    train/images/*.png
    train/annotations_pss/*.png
    test/images/*.png
    test/annotations_pss/*.png
```

Image and part-mask filenames must match. COCO instance JSON provides instance masks; intersecting an instance mask with its aligned semantic part map constructs PIS targets. A separate packed PIS PNG is not required by this mapper. EndoVis part maps use 0=shaft, 1=wrist, 2=clasper, 255=ignore/background; confirm this preprocessing before training. Do not point raw EndoVis palette masks directly at this mapper.

COCO instrument IDs are 1–9 and mapped to contiguous training IDs 0–8: Bipolar, Prograsp, Needle Driver, Scissors, Ultrasound Probe, Suction, Clip Applier, Vessel Sealer, Retractor. This shared vocabulary does not imply every dataset contains every class.

SAR-RARP50 PSS maps use 0=background, 1=shaft, 2=wrist, 3=clasper, 255=ignore. The provided conversion reads raw masks from `annotations/` and writes a separate `annotations_pss/`:

```bash
python tools/prepare_sar_rarp50_pss.py --root "$SAR_RARP50_ROOT"
```

Inspect the conversion table for your downloaded annotation version before applying it. Unsupported raw categories become ignore, not background.

Use explicit, disjoint train splits. Validation/test images must not appear in any training stream. The loader does not perform image-content deduplication for custom datasets.
