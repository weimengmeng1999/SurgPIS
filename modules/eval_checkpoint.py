"""Load one branch of a teacher/student checkpoint for standalone evaluation."""
from collections import OrderedDict
import logging

from detectron2.checkpoint import DetectionCheckpointer


class EvaluationCheckpointer(DetectionCheckpointer):
    def __init__(self, model, *, eval_target, **kwargs):
        super().__init__(model, **kwargs)
        if eval_target not in ('TEACHER', 'STUDENT'):
            raise ValueError(f'Unknown evaluation target: {eval_target}')
        self.eval_target = eval_target

    def _load_model(self, checkpoint):
        weights = checkpoint['model']
        if getattr(self.model, 'explicit_tissue_background', False):
            for key, value in weights.items():
                if key.endswith(('class_embed.weight', 'class_embed.bias')) and value.shape[0] != 11:
                    raise ValueError('Old instrument-only checkpoint cannot initialize the tissue model; '
                                     'use backbone pretraining or an 11-output tissue checkpoint.')
        prefixes = ('modelTeacher.', 'modelStudent.')
        if not any(key.startswith(prefixes) for key in weights):
            if 'sem_seg_head.predictor.class_embed.weight' in weights:
                # Full released models must load strictly, including module-version metadata.
                self.model.load_state_dict(weights, strict=True)
                checkpoint.pop('model')
                logging.getLogger(__name__).info('Strictly loaded standalone model (%d tensors)', len(weights))
                return None
            return super()._load_model(checkpoint)
        branch = 'modelStudent' if self.eval_target == 'STUDENT' else 'modelTeacher'
        prefix = branch + '.'
        selected = OrderedDict((key[len(prefix):], value)
                               for key, value in weights.items() if key.startswith(prefix))
        if not selected:
            raise ValueError(f'Checkpoint has no {branch} weights; cannot evaluate {self.eval_target}')
        # Preserve module versions: MaskFormerHead otherwise applies legacy key conversion.
        if hasattr(weights, '_metadata'):
            selected._metadata = OrderedDict(
                ('' if key == branch else key[len(prefix):], value)
                for key, value in weights._metadata.items()
                if key == branch or key.startswith(prefix))
        self.model.load_state_dict(selected, strict=True)
        checkpoint.pop('model')
        logging.getLogger(__name__).info('Strictly loaded %s (%d tensors)', branch, len(selected))
        return None
