import numpy as np
import torch
from torch import nn
from padertorch.ops.sequence.mask import compute_mask
from padertorch.contrib.je.modules.hybrid import CNN
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.reduce import Mean, Sum, Max
from padertorch.contrib.je.modules.rnn import GRU
from pb_sed.models import base


class CRNN(base.SoundEventModel):
    """
    >>> config = CRNN.get_config({\
            'tag_conditioning': True,\
            'cnn': {\
                'factory': CNN,\
                'cnn_2d': {'out_channels':[32,32,32], 'kernel_size': 3},\
                'cnn_1d': {'out_channels':[32,32], 'kernel_size': 3},\
            },\
            'rnn': {\
                'factory': GRU, 'bidirectional': True, 'hidden_size': 64,\
                'output_net': {'out_channels':[32,10], 'kernel_size': 1}\
            },\
            'feature_extractor': {\
                'sample_rate': 16000,\
                'stft_size': 512,\
                'number_of_filters': 80,\
            },\
        })
    >>> crnn = CRNN.from_config(config)
    >>> inputs = {'stft': torch.randn((4, 1, 5, 257, 2)), 'seq_len': [5,4,3,2], 'weak_targets': torch.zeros((4,10)), 'strong_targets': torch.zeros((4,10,5))}
    >>> outputs = crnn(inputs)
    >>> outputs[0].shape
    torch.Size([4, 10, 5])
    >>> review = crnn.review(inputs, outputs)
    """
    def __init__(
            self, feature_extractor, cnn, rnn, *,
            tag_conditioning=False, labelwise_metrics=(), label_mapping=None,
    ):
        super().__init__(
            labelwise_metrics=labelwise_metrics,
            label_mapping=label_mapping
        )
        self.feature_extractor = feature_extractor
        self.cnn = cnn
        self.rnn = rnn
        self.tag_conditioning = tag_conditioning

    def forward(self, inputs):
        """
        forward used in trainer

        Args:
            inputs: example dict

        Returns:

        """
        if self.training:
            x = inputs.pop('stft')
        else:
            x = inputs['stft']
        seq_len = np.array(inputs['seq_len'])
        if "strong_targets" in inputs:
            weak_targets = inputs["weak_targets"]
            strong_targets = inputs["strong_targets"]
            x, seq_len_x, targets = self.feature_extractor(
                x, seq_len=seq_len, targets=(weak_targets, strong_targets)
            )
        else:
            x, seq_len_x = self.feature_extractor(x, seq_len=seq_len)
            targets = None

        tag_condition = inputs["tag_condition"].unsqueeze(-1) if self.tag_conditioning else None
        h, seq_len_h = self.cnn(x, seq_len_x, tag_condition)
        if self.tag_conditioning:
            b, f, t = h.shape
            tag_condition = torch.broadcast_to(tag_condition, (b, tag_condition.shape[1], t))
            h = torch.cat([h, tag_condition], dim=1)

        y, seq_len_y = self.rnn(h, seq_len_h)
        return nn.Sigmoid()(y), seq_len_y, x, seq_len_x, targets

    def review(self, inputs, outputs):
        """
        compute loss and metrics

        Args:
            inputs:
            outputs:

        Returns:

        """
        y, seq_len_y, x, seq_len_x, targets = outputs
        assert targets is not None
        weak_targets, strong_targets = targets
        assert strong_targets.shape == y.shape, (strong_targets.shape, y.shape)
        strong_targets_mask = (strong_targets > .99) + (strong_targets < .01)
        bce = nn.BCELoss(reduction='none')(y, strong_targets) * strong_targets_mask
        bce = Sum(axis=-1)(bce, seq_len_y).sum() / strong_targets_mask.sum()

        strongly_labeled_examples_idx = np.argwhere((Mean(axis=-1)(strong_targets_mask, seq_len_y) > .999).detach().cpu().numpy().all(-1)).flatten()
        y = y.detach().cpu().numpy()
        targets = strong_targets.detach().cpu().numpy()
        review = dict(
            loss=bce,
            scalars=dict(
                seq_len=np.mean(inputs['seq_len']),
                strong_label_rate=strong_targets_mask.detach().cpu().numpy().mean(),
            ),
            images=dict(
                features=x[:3],
                strong_targets=strong_targets[:3],
            ),
            buffers=dict(
                y_strong=np.concatenate([y[i, :, :seq_len_y[i]].T for i in strongly_labeled_examples_idx]),
                targets_strong=np.concatenate([targets[i, :, :seq_len_y[i]].T for i in strongly_labeled_examples_idx]),
            )
        )
        return review

    def modify_summary(self, summary):
        """called by the trainer before dumping a summary

        Args:
            summary:

        Returns:

        """
        if f'targets_strong' in summary['buffers']:
            # Computes fscores from scores and targets
            self.add_metrics_to_summary(summary, 'strong')
        summary = super().modify_summary(summary)
        return summary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """Automatically prepares/completes the configuration of the model.

        You do not need to understand how this is working as there is a lot of
        magic in the background which serves convenience and is not crucial to
        run the model.

        Args:
            config:

        Returns:

        """
        config['feature_extractor'] = {'factory': NormalizedLogMelExtractor}
        config['cnn'] = {'factory': CNN}
        config['rnn'] = {'factory': GRU}
        input_size = config['feature_extractor']['number_of_filters']
        num_events = config['rnn']['output_net']['out_channels'][-1]
        in_channels = (
            1 + config['feature_extractor']['add_deltas']
            + config['feature_extractor']['add_delta_deltas']
            + config['cnn']['positional_encoding']
        )
        if config['tag_conditioning']:
            config['cnn']['conditional_dims'] = num_events
            in_channels += num_events
        config['cnn']['cnn_2d']['in_channels'] = in_channels
        config['cnn']['input_height'] = input_size
        input_size = config['cnn']['cnn_1d']['out_channels'][-1]
        if config['tag_conditioning']:
            input_size += num_events

        if config['rnn']['factory'] == GRU:
            config['rnn'].update({
                'num_layers': 1,
                'bias': True,
                'dropout': 0.,
                'bidirectional': True
            })

        if input_size is not None:
            config['rnn']['input_size'] = input_size

    def tagging(self, inputs):
        y, seq_len_y, *_ = self.forward(inputs)
        return Max(-1, keepdims=True)(y)[0], np.ones_like(seq_len_y)

    def boundaries_detection(self, inputs):
        return self.sound_event_detection(inputs)

    def sound_event_detection(self, inputs):
        y, seq_len_y, *_ = self.forward(inputs)
        seq_mask = compute_mask(y, seq_len_y, batch_axis=0, sequence_axis=-1)
        return y*seq_mask, seq_len_y


def tune_tagging(
        crnns, dataset, device, timestamps, event_classes, metrics,
        minimize=False, storage_dir=None
):
    print()
    print('Tagging Tuning')
    tagging_scores = base.tagging(
        crnns, dataset, device,
        timestamps=timestamps, event_classes=event_classes,
    )
    return base.tune_tagging(
        tagging_scores, medfilt_length_candidates=[1],
        metrics=metrics, minimize=minimize, storage_dir=storage_dir,
    )


def tune_boundary_detection(
        crnns, dataset, device, timestamps, event_classes, tags, metrics,
        stepfilt_lengths, minimize=False, tag_masking=True, storage_dir=None,
):
    print()
    print('Boundaries Detection Tuning')
    boundaries_scores = base.boundaries_detection(
        crnns, dataset, device,
        stepfilt_length=None, apply_mask=False, masks=tags,
        timestamps=timestamps, event_classes=event_classes,
    )
    return base.tune_boundaries_detection(
        boundaries_scores, medfilt_length_candidates=[1],
        stepfilt_length_candidates=stepfilt_lengths,
        tags=tags, metrics=metrics, minimize=minimize,
        tag_masking=tag_masking, storage_dir=storage_dir,
    )


def tune_sound_event_detection(
        crnns, dataset, device, timestamps, event_classes, tags, metrics,
        medfilt_lengths, minimize=False, tag_masking='?', storage_dir=None,
):
    print()
    print('Sound Event Detection Tuning')
    detection_scores = base.sound_event_detection(
        crnns, dataset, device,
        timestamps=timestamps, event_classes=event_classes,
    )
    return base.tune_sound_event_detection(
        detection_scores, medfilt_lengths, tags,
        metrics=metrics, minimize=minimize,
        tag_masking=tag_masking, storage_dir=storage_dir,
    )
