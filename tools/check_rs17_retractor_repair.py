import sys,json,hashlib,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
import mask2former
from detectron2.data import DatasetCatalog,MetadataCatalog,transforms as T
from detectron2.data.datasets.coco import load_coco_json
from mask2former.data.datasets.register_endovis17_retractor import repaired_records
from mask2former.data.datasets.rs17_retractor_repair import decode
from mask2former.data.dataset_mappers.mask_former_inspart_dataset_mapper import MaskFormerInspartDatasetMapper
root=Path(MetadataCatalog.get('endovis17_inp_train').image_root).parent.parent
paths=[root/'endovis_instance_train.json',root/'endovis_instance_val.json']
original_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
raw=load_coco_json(str(paths[0]),str(root/'train/images'));new=repaired_records();changes=[]
for a,b in zip(raw,new):
    if 'rs17_retractor_repair' not in b:
        assert a==b;continue
    name=Path(a['file_name']).name; palette=np.array(Image.open(root/'train/annotations'/name))
    info=b['rs17_retractor_repair'];fragments=np.isin(palette,info['source_ids'])
    oldmasks=[decode(x['segmentation'],a['height'],a['width']) for x in a['annotations']]
    kept=[(x,m) for x,m in zip(a['annotations'],oldmasks) if not (m & fragments).any()]
    assert len(kept)==len(b['annotations'])
    newmasks=[]
    for (old,oldmask),fixed in zip(kept,b['annotations']):
        mask=decode(fixed['segmentation'],a['height'],a['width']);newmasks.append(mask)
        assert old['category_id']==fixed['category_id']
        np.testing.assert_array_equal(oldmask & ~fragments,mask & ~fragments)
        assert np.all(mask[oldmask])
    union=np.sum(newmasks,axis=0);assert np.all(union<=1)
    np.testing.assert_array_equal(union>0,np.any(oldmasks,axis=0)|fragments)
    changes.append(dict(file=name,**info))
counts=lambda rows:collections.Counter(a['category_id'] for r in rows for a in r['annotations'])
assert counts(raw)[9]==130 and counts(new)[9]==53
assert sum(c['removed_targets'] for c in changes)==77
assert {k:v for k,v in counts(raw).items() if k!=9}=={k:v for k,v in counts(new).items() if k!=9}
pis=DatasetCatalog.get('endovis17_inp_train_retractorfix');iis=DatasetCatalog.get('endovis17_iis_train_retractorfix')
byid={r['image_id']:r for r in iis};fixed={r['file_name']:r for r in new}
for r in pis:
    assert r['instances']==fixed[r['file_name']]['annotations']
    assert [a['category_id']-1 for a in r['instances']]==[a['category_id'] for a in byid[r['image_id']]['annotations']]
    assert [a['segmentation'] for a in r['instances']]==[a['segmentation'] for a in byid[r['image_id']]['annotations']]
torch.set_num_threads(2)
mapper=MaskFormerInspartDatasetMapper(is_train=True,augmentations=[T.Resize((512,512))],image_format='BGR',size_divisibility=512,ignore_label=255,meta=MetadataCatalog.get('endovis17_inp_train_retractorfix'))
examples={'seq_3_frame184','seq_5_frame010','seq_6_frame033','seq_7_frame029','seq_16_frame288'}
for r in pis:
    if r['image_id'] not in examples:continue
    out=mapper(r);inst=out['instances']
    for mask,parts in zip(inst.gt_masks,inst.gt_masks_parts):
        for pid in range(3):assert torch.equal(parts[pid],mask.bool() & (out['sem_seg']==pid))
assert original_hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
# Original validation is still the only validation catalog used by this ablation.
assert sum(a['category_id']==9 for r in load_coco_json(str(paths[1]),str(root/'val/images')) for a in r['annotations'])==175
report=dict(frames=len(new),changed_frames=len(changes),original_counts=dict(counts(raw)),repaired_counts=dict(counts(new)),pixels=sum(c['pixels'] for c in changes),max_assignment_distance_px=max(c['max_assignment_distance_px'] for c in changes),changes=changes,source_json_sha256=original_hashes,validation_retractors=175,mapper_checks='PASS')
output=Path(sys.argv[1]);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['changes','source_json_sha256']},indent=2),flush=True)
