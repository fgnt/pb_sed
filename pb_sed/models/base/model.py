import abc
import numpy as np
from padertorch import Model
from torchvision.utils import make_grid
from pb_sed.evaluation import instance_based
from sklearn import metrics


class SoundEventModel(Model, abc.ABC):
    def __init__(self, *, labelwise_metrics=(), label_mapping=None, test_labels=None):
        super().__init__()
        self.labelwise_metrics = labelwise_metrics
        self.label_mapping = label_mapping
        self.test_labels = test_labels

    @abc.abstractmethod
    def tagging(self, inputs, **params):
        pass

    @abc.abstractmethod
    def boundaries_detection(self, inputs, **params):
        pass

    @abc.abstractmethod
    def sound_event_detection(self, inputs, **params):
        pass

    def modify_summary(self, summary):
        for key, scalar in summary['scalars'].items():
            # average scalar metrics over batches
            summary['scalars'][key] = np.mean(scalar)

        for key, image in summary['images'].items():
            # prepare image grid for tensorboard
            if image.dim() == 4 and image.shape[1] > 1:
                image = image[:, 0]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary

    def add_metrics_to_summary(self, summary, suffix):
        y = np.concatenate(summary['buffers'].pop(f'y_{suffix}'))
        if y is None:
            return

        summary['scalars'][f'num_examples_{suffix}'] = len(y)
        targets = np.concatenate(summary['buffers'].pop(f'targets_{suffix}'))

        test_labels = self.test_labels
        if test_labels is not None:
            if isinstance(test_labels[0], str):
                assert self.label_mapping is not None
                test_labels = [self.label_mapping.index(label) for label in test_labels]
            y = y[..., test_labels]
            targets = targets[..., test_labels]

        def maybe_add_label_wise(key, values):
            if key in self.labelwise_metrics:
                for event_class, value in enumerate(values):
                    if test_labels is not None:
                        event_class = test_labels[event_class]
                    if self.label_mapping is not None:
                        event_class = self.label_mapping[event_class]
                    summary['scalars'][f'z/{key}/{event_class}'] = value

        _, f, p, r = instance_based.get_best_fscore_thresholds(targets, y)
        summary['scalars'][f'macro_fscore_{suffix}'] = f.mean()
        maybe_add_label_wise(f'fscore_{suffix}', f)

        _, er, ir, dr = instance_based.get_best_er_thresholds(targets, y)
        summary['scalars'][f'macro_error_rate_{suffix}'] = er.mean()
        maybe_add_label_wise(f'error_rate_{suffix}', er)

        lwlrap, per_class_lwlrap, weight_per_class = instance_based.lwlrap(targets, y)
        summary['scalars'][f'lwlrap_{suffix}'] = lwlrap
        maybe_add_label_wise(f'lwlrap_{suffix}', per_class_lwlrap)

        if (targets.sum(0) > 1).all():
            ap = metrics.average_precision_score(targets, y, average=None)
            summary['scalars'][f'map_{suffix}'] = np.mean(ap)
            maybe_add_label_wise(f'ap_{suffix}', ap)

            auc = metrics.roc_auc_score(targets, y, average=None)
            summary['scalars'][f'mauc_{suffix}'] = np.mean(auc)
            maybe_add_label_wise(f'auc_{suffix}', auc)
