"""Compare existing scoring with pinned TAPPS code, without changing model/evaluator.

Run with the project Python environment. Class offset -1 is for legacy checkpoints.
"""
import argparse
import ast
from collections import defaultdict, OrderedDict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    def confusion_matrix(gt, pred):
        labels = np.unique(np.concatenate([gt, pred]))
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        np.add.at(matrix, (np.searchsorted(labels, gt), np.searchsorted(labels, pred)), 1)
        return matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_net import Trainer, setup
from detectron2.engine import default_argument_parser
from mask2former.evaluation.part_pq import PQStat, part_pq_stats
from mask2former.evaluation.part_pq_evaluator import PartPQEvaluator, _map_global_part_id_to_generic


def reference_functions(path):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {'PQStatCat', 'PQStat', 'pq_part'}]
    ns = {'np': np, 'defaultdict': defaultdict, 'confusion_matrix': confusion_matrix}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)
    # Same default sklearn confusion matrix for the adapter's integer labels
    # 0..255. Bincount avoids repeated sorting of full-resolution pixel arrays.
    def fast_confusion_matrix(gt, pred):
        gt, pred = np.asarray(gt, dtype=np.int64), np.asarray(pred, dtype=np.int64)
        assert np.all((gt >= 0) & (gt <= 255)) and np.all((pred >= 0) & (pred <= 255))
        cm = np.bincount(gt * 256 + pred, minlength=256 * 256).reshape(256, 256)
        labels = np.flatnonzero(cm.sum(0) + cm.sum(1))
        return cm[np.ix_(labels, labels)]
    rng = np.random.default_rng(17)
    for _ in range(20):
        gt = rng.choice([0, 1, 2, 3, 255], 1000)
        pred = rng.choice([0, 1, 2, 3, 255], 1000)
        np.testing.assert_array_equal(fast_confusion_matrix(gt, pred), confusion_matrix(gt, pred))
    ns['confusion_matrix'] = fast_confusion_matrix
    return ns


def to_reference(gt, pred, part_categories, shape):
    """Explicit EndoVis adapter: valid image background, no invented void region.

    Generic parts 0..2 become reference parts 1..3; ignored GT becomes 0.
    A part-bearing category's wholly unlabeled GT instance becomes crowd,
    as in reference annotation_parsing. Predicted missing parts become 255.
    """
    gm, pm, cm = {}, {}, {}
    for index, category in enumerate(part_categories):
        crowd = np.zeros(shape, dtype=bool)
        groups = []
        for instances, is_gt in [(gt, True), (pred, False)]:
            masks, parts = [], []
            for inst in instances:
                if inst['category_id'] != category:
                    continue
                mask, part = inst['mask'], inst['part_map']
                if is_gt and not (mask & (part != 255)).any():
                    crowd |= mask
                    continue
                annotation = np.zeros(shape, dtype=np.int32)
                annotation[mask] = np.where(part[mask] == 255, 0 if is_gt else 255, part[mask] + 1)
                masks.append(mask)
                parts.append(annotation)
            groups.append({'num_instances': len(masks),
                           'binary_masks': np.array(masks).reshape((-1,) + shape),
                           'parts_annotation': np.array(parts).reshape((-1,) + shape)})
        gm[index], pm[index] = groups
        cm[index] = {'void_masks': np.zeros(shape, dtype=bool), 'crowd_masks': crowd}
    definition = {'num_cats': len(part_categories),
                  'cat_def': [{'sem_cls': [c], 'parts_cls': [1, 2, 3]} for c in part_categories]}
    return pm, gm, cm, definition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True)
    ap.add_argument('--datasets', nargs='+', default=['endovis_inp_val', 'endovis17_inp_val'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config-file', default='configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k_partpq.yaml')
    ap.add_argument('--model', choices=['TEACHER', 'STUDENT'], default='TEACHER')
    ap.add_argument('--offsets', type=int, nargs='+', default=[0])
    ap.add_argument('--output-dir')
    args = ap.parse_args()
    ref = reference_functions(args.reference)
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir) if args.output_dir else root / 'output/partpq'
    out.mkdir(parents=True, exist_ok=True)
    cli = default_argument_parser().parse_args([
        '--config-file', args.config_file,
        '--eval-only', 'OUTPUT_DIR', str(out), 'SAVE_PREDICTIONS', 'False',
        'DATALOADER.NUM_WORKERS', '0', 'WSL.EVAL_WHO', args.model,
        'MODEL.WEIGHTS', str(args.checkpoint),
        'MODEL.MASK_FORMER.TEST.INSTANCE_ON', 'True',
        'MODEL.MASK_FORMER.TEST.PANOPTIC_ON', 'True'])
    cfg = setup(cli)
    model = Trainer.build_model(cfg)
    checkpoint = Path(args.checkpoint)
    from modules.eval_checkpoint import EvaluationCheckpointer
    state = EvaluationCheckpointer(model, eval_target=args.model).load(str(checkpoint))
    iteration = state.get('iteration')
    del state
    model.eval()
    print(f'Checkpoint loaded strictly: {args.model}, iteration {iteration}', flush=True)
    for dataset in args.datasets:
        ev = PartPQEvaluator(dataset, distributed=False)
        cats = set()
        for image_id in ev._gt_by_image_id:
            for g in ev._load_gt_instances(image_id):
                if (g['mask'] & (g['part_map'] != 255)).any(): cats.add(g['category_id'])
        cats = sorted(cats)
        current = {offset: PQStat() for offset in args.offsets}
        pan_local = {offset: PQStat() for offset in args.offsets}
        pan_ref = {offset: ref['PQStat']() for offset in args.offsets}
        start = time.time()
        for count, inputs in enumerate(Trainer.build_test_loader(cfg, dataset), 1):
            with torch.no_grad(): output = model(inputs)[0]
            gt = ev._load_gt_instances(inputs[0]['image_id'])
            part = _map_global_part_id_to_generic(output['parts'].cpu().numpy())
            instances = output['instances'].to('cpu')
            keep = instances.scores.numpy() > .5
            hybrid = [{'category_id': int(c), 'mask': m.astype(bool), 'part_map': part}
                      for c, m in zip(instances.pred_classes.numpy()[keep], instances.pred_masks.numpy()[keep])]
            pan_map, segs = output['panoptic_seg']
            pan_map = pan_map.cpu().numpy()
            pan = [{'category_id': s['category_id'], 'mask': pan_map == s['id'], 'part_map': part} for s in segs]
            for offset in args.offsets:
                def remap(items):
                    return [dict(i, category_id=i['category_id'] + offset) for i in items
                            if i['category_id'] + offset in ev._valid_category_ids]
                hp, pp = remap(hybrid), remap(pan)
                current[offset] += part_pq_stats(gt, hp, 3)
                pan_local[offset] += part_pq_stats(gt, pp, 3)
                pm, gm, cm, definition = to_reference(gt, pp, cats, part.shape)
                pan_ref[offset] += ref['pq_part'](pm, gm, cm, definition, 255)
            if count % 25 == 0:
                print(dataset, count, 'elapsed', round(time.time()-start), flush=True)
            if args.limit and count >= args.limit: break
        result = {'dataset': dataset, 'images': count, 'part_category_ids_zero_based': cats,
                  'checkpoint': str(checkpoint), 'model': args.model, 'iteration': iteration, 'object_threshold': .8, 'instance_threshold': .5,
                  'reference_source': args.reference, 'variants': {}}
        for offset in args.offsets:
            result['variants'][str(offset)] = {
                'current_hybrid': current[offset].average(),
                'panoptic_local_core': pan_local[offset].average(),
                'panoptic_tapps_core': pan_ref[offset].pq_average(definition)}
        target = out / (dataset + '_reference_comparison.json')
        target.write_text(json.dumps(result, indent=2))
        print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
