"""Download the public Stage-1 RN50 weights and evaluate the two validation splits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = 'oliviamw/SurgPIS-Stage1-RN50'
REVISION = 'a6c649dc3d7c5d05c1573330012aec2f30e0adb5'
FILENAME = 'surgpis_stage1_rn50.pth'
SHA256 = '902f406ece382acfe7de5e005ae5a4ad2793a0571f2333ea853d77d653241b6f'
EXPECTED = {'endovis17_inp_val': (600, 52.20751508992066),
            'endovis_inp_val': (596, 81.83298309123475)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,help='Optional already-downloaded official weight file')
    ap.add_argument('--output-dir',type=Path,default=Path('output/stage1_pretrained'))
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.checkpoint:
        checkpoint=args.checkpoint.resolve()
    else:
        from huggingface_hub import hf_hub_download
        checkpoint=Path(hf_hub_download(REPO,FILENAME,revision=REVISION))
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError('Checkpoint SHA256 does not match the published Stage-1 RN50 weights')
    subprocess.run([sys.executable,str(root/'tools/audit_partpq_tapps.py'),
        '--config-file',str(root/'configs/endovis/instance-part-segmentation/stage1_R50_pretrained.yaml'),
        '--reference',str(root/'tools/reference_tapps/evaluation_PartPQ.py'),
        '--checkpoint',str(checkpoint),'--model','TEACHER','--offsets','0',
        '--datasets',*EXPECTED,'--output-dir',str(out)],cwd=root,check=True)
    scores={}
    for dataset,(count,expected) in EXPECTED.items():
        result=json.loads((out/(dataset+'_reference_comparison.json')).read_text())
        assert result['images']==count, (dataset,result['images'],count)
        score=100*result['variants']['0']['panoptic_tapps_core'][0][1]['PartPQ_parts']
        scores[dataset]=dict(images=count,partpq_percent=score,reference_percent=expected,
                             delta_percentage_points=score-expected,within_0_01pp=abs(score-expected)<=.01)
    report=dict(repo=REPO,revision=REVISION,checkpoint_sha256=digest,scores=scores)
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not all(s['within_0_01pp'] for s in scores.values()):
        raise RuntimeError('Scores differ from reference; check data annotations and software environment')

if __name__=='__main__':main()
