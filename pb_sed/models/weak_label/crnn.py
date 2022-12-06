import numpy as np
import torch
from torch import nn
from einops import rearrange
from padertorch.ops.sequence.mask import compute_mask
from padertorch.contrib.je.modules.conv import Pad
from padertorch.contrib.je.modules.hybrid import CNN
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.reduce import TakeLast, Mean
from padertorch.contrib.je.modules.rnn import GRU, TransformerEncoder
from pb_sed.models import base


class CRNN(base.SoundEventModel):
    """
    >>> config = CRNN.get_config({\
            'cnn': {\
                'factory': CNN,\
                'cnn_2d': {'out_channels':[32,32,32], 'kernel_size': 3},\
                'cnn_1d': {'out_channels':[32,32], 'kernel_size': 3},\
            },\
            'rnn_fwd': {'factory': GRU, 'rnn': {'hidden_size': 64}, 'output_net': {'out_channels':[32,10], 'kernel_size': 1}},\
            'feature_extractor': {\
                'sample_rate': 16000,\
                'stft_size': 512,\
                'number_of_filters': 80,\
            },\
        })
    >>> crnn = CRNN.from_config(config)
    >>> np.random.seed(3)
    >>> inputs = {'stft': torch.tensor(np.random.randn(4, 1, 15, 257, 2), dtype=torch.float32), 'seq_len': [15, 14, 13, 12], 'weak_targets': torch.zeros((4,10)), 'boundary_targets': torch.zeros((4,10,15))}
    >>> outputs = crnn({**inputs})
    >>> outputs[0].shape
    torch.Size([4, 10, 15])
    >>> review = crnn.review(inputs, outputs)
    """
    def __init__(
            self, feature_extractor, cnn, rnn_fwd, rnn_bwd,
            *, minimum_score=1e-5, label_smoothing=0.,
            labelwise_metrics=(), label_mapping=None, test_labels=None,
            slat=False, strong_fwd_bwd_loss_weight=1., class_weights=None,
    ):
        super().__init__(
            labelwise_metrics=labelwise_metrics,
            label_mapping=label_mapping,
            test_labels=test_labels,
        )
        self.feature_extractor = feature_extractor
        self.cnn = cnn
        self.rnn_fwd = rnn_fwd
        self.rnn_bwd = rnn_bwd
        self.minimum_score = minimum_score
        self.label_smoothing = label_smoothing
        self.slat = slat
        self.strong_fwd_bwd_loss_weight = strong_fwd_bwd_loss_weight
        self.class_weights = None if class_weights is None else torch.Tensor(class_weights)

    def sigmoid(self, y):
        return self.minimum_score + (1-2*self.minimum_score) * nn.Sigmoid()(y)

    def fwd_tagging(self, h, seq_len):
        y, seq_len_y = self.rnn_fwd(h, seq_len)
        return self.sigmoid(y), seq_len_y

    def bwd_tagging(self, h, seq_len):
        y, seq_len_y = self.rnn_bwd(h, seq_len)
        return self.sigmoid(y), seq_len_y

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
        if "weak_targets" in inputs:
            targets = self.read_targets(inputs)
            x, seq_len_x, targets = self.feature_extractor(
                x, seq_len=seq_len, targets=targets
            )
        else:
            x, seq_len_x = self.feature_extractor(x, seq_len=seq_len)
            targets = None

        h, seq_len_h = self.cnn(x, seq_len_x)
        y_fwd, seq_len_y = self.fwd_tagging(h, seq_len_h)
        if self.rnn_bwd is None:
            y_bwd = None
        else:
            y_bwd, seq_len_y_ = self.bwd_tagging(h, seq_len_h)
            assert (seq_len_y_ == seq_len_y).all()
        return y_fwd, y_bwd, seq_len_y, x, seq_len_x, targets

    def read_targets(self, inputs, subsample_idx=None):
        if 'boundary_targets' in inputs:
            return inputs['weak_targets'], inputs['boundary_targets']
        return inputs['weak_targets'],

    def review(self, inputs, outputs):
        """compute loss and metrics

        Args:
            inputs:
            outputs:

        Returns:

        """
        y_fwd, y_bwd, seq_len, x, _, targets = outputs
        assert targets is not None
        weak_targets = targets[0]
        weak_targets_mask = (weak_targets < .01) + (weak_targets > .99)
        weak_targets = weak_targets * weak_targets_mask
        weak_label_rate = weak_targets_mask.detach().cpu().numpy().mean()

        loss = (
            self.compute_weak_fwd_bwd_loss(y_fwd, y_bwd, weak_targets, seq_len)
            * weak_targets_mask[..., None]
        )

        if self.strong_fwd_bwd_loss_weight > 0.:
            if self.slat:
                boundary_targets = weak_targets[..., None].expand(y_fwd.shape)
            else:
                assert len(targets) == 2, len(targets)
                boundary_targets = targets[1]
            boundary_targets_mask = (boundary_targets > .99) + (boundary_targets < .01)
            boundary_targets_mask = boundary_targets_mask * (boundary_targets_mask.float().mean(-1, keepdim=True) > .999) * (weak_targets > .99)[..., None]
            boundary_label_rate = boundary_targets_mask.detach().cpu().numpy().mean()
            if (boundary_targets_mask == 1).any():
                strong_label_loss = self.compute_strong_fwd_bwd_loss(
                y_fwd, y_bwd, boundary_targets)
                strong_fwd_bwd_loss_weight = (
                    boundary_targets_mask * self.strong_fwd_bwd_loss_weight)
                loss = strong_fwd_bwd_loss_weight * strong_label_loss + (1. - strong_fwd_bwd_loss_weight) * loss
        else:
            boundary_label_rate = 0.

        loss = Mean(axis=-1)(loss, seq_len)
        if self.class_weights is None:
            weights = weak_targets_mask
        else:
            self.class_weights = self.class_weights.to(loss.device)
            weights = weak_targets_mask * self.class_weights
        loss = (loss * weights).sum() / weights.sum()

        labeled_examples_idx = (
            weak_targets_mask.detach().cpu().numpy() == 1
        ).all(-1)
        y_weak = TakeLast(axis=2)(y_fwd, seq_len=seq_len)
        if y_bwd is not None:
            y_weak = y_weak / 2 + y_bwd[..., 0] / 2
        y_weak = y_weak.detach().cpu().numpy()[labeled_examples_idx]
        weak_targets = weak_targets.detach().cpu().numpy()[labeled_examples_idx]
        review = dict(
            loss=loss,
            scalars=dict(
                seq_len=np.mean(inputs['seq_len']),
                weak_label_rate=weak_label_rate,
                boundary_label_rate=boundary_label_rate,
            ),
            images=dict(
                features=x[:3],
            ),
            buffers=dict(
                y_weak=y_weak,
                targets_weak=weak_targets,
            ),
        )
        return review

    def compute_weak_fwd_bwd_loss(self, y_fwd, y_bwd, targets, seq_len):
        if self.label_smoothing > 0.:
            targets = torch.clip(
                targets, min=self.label_smoothing, max=1-self.label_smoothing)
        if y_bwd is None:
            y_weak = TakeLast(axis=2)(y_fwd, seq_len=seq_len)
            # y_weak = y_weak + 0.1 * (weak_targets - y_weak)
            return nn.BCELoss(reduction='none')(y_weak, targets)[..., None].expand(y_fwd.shape)
        else:
            y_weak = torch.maximum(y_fwd, y_bwd)
            targets = targets[..., None].expand(y_weak.shape)
            # y_weak = y_weak + 0.1 * (weak_targets_ - y_weak)
            return nn.BCELoss(reduction='none')(y_weak, targets)

    def compute_strong_fwd_bwd_loss(self, y_fwd, y_bwd, targets):
        if self.label_smoothing > 0.:
            targets = torch.clip(
                targets, min=self.label_smoothing, max=1-self.label_smoothing)
        strong_targets_fwd = torch.cummax(targets, dim=-1)[0]
        strong_targets_bwd = torch.cummax(targets.flip(-1), dim=-1)[0].flip(-1)
        loss = nn.BCELoss(reduction='none')(y_fwd, strong_targets_fwd)
        if y_bwd is not None:
            loss = (
                loss/2
                + nn.BCELoss(reduction='none')(y_bwd, strong_targets_bwd)/2
            )
        return loss

    def modify_summary(self, summary):
        """called by the trainer before dumping a summary

        Args:
            summary:

        Returns:

        """
        if f'targets_weak' in summary['buffers']:
            # Computes fscores from scores and targets
            self.add_metrics_to_summary(summary, 'weak')
        summary = super().modify_summary(summary)
        return summary

    def tagging(self, inputs):
        y_fwd, y_bwd, seq_len_y, *_ = self.forward(inputs)
        seq_len = np.ones_like(seq_len_y)
        if y_bwd is None:
            return TakeLast(axis=-1, keepdims=True)(y_fwd, seq_len_y), seq_len
        return (
            (
                TakeLast(axis=-1, keepdims=True)(y_fwd, seq_len_y)
                + y_bwd[..., :1]
            ) / 2,
            seq_len
        )

    def boundaries_detection(self, inputs):
        y_fwd, y_bwd, seq_len_y, *_ = self.forward(inputs)
        seq_mask = compute_mask(y_fwd, seq_len_y, batch_axis=0, sequence_axis=-1)
        return torch.minimum(y_fwd*seq_mask, y_bwd*seq_mask), seq_len_y

    def sound_event_detection(self, inputs, window_length, window_shift=1):
        """SED by applying the model to small segments around each frame

        Args:
            inputs:
            window_length:
            window_shift:

        Returns:

        """
        window_length = np.array(window_length, dtype=np.int)
        x = inputs['stft']
        seq_len = np.array(inputs['seq_len'])
        x, seq_len = self.feature_extractor(x, seq_len=seq_len)
        h, seq_len = self.cnn(x, seq_len)
        if window_length.ndim == 0:
            return self._single_window_length_sed(
                h, seq_len, window_length, window_shift
            )
        window_lengths_flat = np.unique(window_length.flatten())
        y = None
        for i, win_len in enumerate(window_lengths_flat):
            yi, seq_len_y = self._single_window_length_sed(
                h, seq_len, win_len, window_shift
            )
            b, k, t = yi.shape
            if window_length.ndim == 1:
                assert window_length.shape[0] in [1, k], window_length.shape
            elif window_length.ndim == 2:
                assert window_length.shape[1] in [1, k], window_length.shape
                n = window_length.shape[0]
                window_length = np.broadcast_to(window_length, (n, k))
                yi = yi[:, None]
            else:
                raise ValueError(
                    'window_length.ndim must not be greater than 2.')
            if y is None:
                y = torch.zeros((b, *window_length.shape, t), device=yi.device)
            mask = torch.from_numpy(window_length.copy()).to(yi.device) == win_len
            y += mask[..., None] * yi
        return y, seq_len_y

    def _single_window_length_sed(
            self, h, seq_len, window_length, window_shift
    ):
        b, f, t = h.shape
        if window_length > window_shift:
            h = Pad('both')(h, (window_length - window_shift))
        h = Pad('end')(h, window_shift - 1)
        indices = np.arange(0, t, window_shift)
        h = [h[..., i:i + window_length] for i in indices]
        n = len(h)
        h = torch.cat(h, dim=0)
        y, _ = self.fwd_tagging(h, seq_len=None)
        y = rearrange(y[..., -1], '(n b) k -> b k n', b=b, n=n)
        if self.rnn_bwd is not None:
            y_bwd, _ = self.bwd_tagging(h, seq_len=None)
            y_bwd = rearrange(y_bwd[..., 0], '(n b) k -> b k n', b=b, n=n)
            y = (y + y_bwd) / 2
        seq_len = 1 + (seq_len-1) // window_shift
        return y, seq_len

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
        config['rnn_fwd'] = {'factory': GRU}
        config['rnn_bwd'] = {}
        input_size = config['feature_extractor']['number_of_filters']
        in_channels = (
            1 + config['feature_extractor']['add_deltas']
            + config['feature_extractor']['add_delta_deltas']
            + config['cnn']['positional_encoding']
        )
        config['cnn']['cnn_2d']['in_channels'] = in_channels
        config['cnn']['input_height'] = input_size
        input_size = config['cnn']['cnn_1d']['out_channels'][-1]

        if input_size is not None:
            if config['rnn_fwd']['rnn'] is None:
                config['rnn_fwd']['output_net']['in_channels'] = input_size
            else:
                config['rnn_fwd']['rnn']['input_size'] = input_size

        if config['rnn_bwd'] is not None:
            # assert config['rnn_bwd']['factory'] == config['rnn_fwd']['factory'], (config['rnn_fwd']['factory'], config['rnn_bwd']['factory'])
            config['rnn_bwd'].update(config['rnn_fwd'].to_dict(), reverse=True)


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
        stepfilt_lengths, minimize=False, tag_masking='?', storage_dir=None,
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
        tag_masking=tag_masking,
        storage_dir=storage_dir,
    )


def tune_sound_event_detection(
        crnns, dataset, device, timestamps, event_classes, tags, metrics,
        window_lengths, window_shift, medfilt_lengths,
        minimize=False, tag_masking='?', storage_dir=None,
):
    print()
    print('Sound Event Detection Tuning')
    leaderboard = {}
    for win_len in window_lengths:
        print()
        print(f'### window_length={win_len} ###')
        detection_scores = base.sound_event_detection(
            crnns, dataset, device,
            model_kwargs={
                'window_length': win_len, 'window_shift': window_shift
            },
            timestamps=timestamps[::window_shift], event_classes=event_classes,
        )
        leaderboard_for_winlen = base.tune_sound_event_detection(
            detection_scores, medfilt_lengths, tags,
            metrics=metrics, minimize=minimize,
            tag_masking=tag_masking,
            storage_dir=storage_dir,
        )
        for metric_name in leaderboard_for_winlen:
            metric_values = leaderboard_for_winlen[metric_name][0]
            hyper_params = leaderboard_for_winlen[metric_name][1]
            scores = leaderboard_for_winlen[metric_name][2]
            for event_class in event_classes:
                hyper_params[event_class]['window_length'] = win_len
                hyper_params[event_class]['window_shift'] = window_shift
            leaderboard = base.update_leaderboard(
                leaderboard, metric_name, metric_values, hyper_params, scores,
                minimize=minimize,
            )
    print()
    print('best overall:')
    for metric_name in metrics:
        print()
        print(metric_name, ':')
        print(leaderboard[metric_name][0])

    return leaderboard
