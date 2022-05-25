
import numpy as np
import dataclasses
from typing import Callable
from padertorch.contrib.je.data.transforms import (
    STFT, TimeWarpedSTFT, MultiHotAlignmentEncoder
)
from pb_sed.data_preparation.utils import add_label_types


@dataclasses.dataclass
class Transform:
    stft: STFT
    label_encoder: MultiHotAlignmentEncoder
    # augmentation:
    anchor_sampling_fn: Callable = None
    anchor_shift_sampling_fn: Callable = None

    def __post_init__(self):
        if isinstance(self.stft, dict):
            self.stft = STFT(**self.stft)
        assert isinstance(self.stft, STFT), type(self.stft)

    def __call__(self, example):
        """
        >>> mix_fn = Transform(min_overlap=1.)
        >>> example1 = {'example_id': '0', 'dataset': '0', 'audio_data': np.zeros((1, 16000)), 'events': ['a', 'a'], 'events_start_samples': [2000,12000], 'events_stop_samples': [8000,14000], 'label_types': ['strong','strong'],}
        >>> example2 = {'example_id': '1', 'dataset': '1', 'audio_data': np.zeros((1, 16000)), 'events': ['a', 'b'], 'events_start_samples': [0, 1000], 'events_stop_samples': [16000, 4000], 'label_types': ['weak', 'strong']}
        >>> ex = mix_fn([example1, example2])
        >>> stft = STFT(200, 801, alignment_keys=['events'], pad=False, fading='half')
        >>> label_enc = MultiHotAlignmentEncoder('events')
        >>> label_enc.initialize_labels(['a','b'])
        >>> transform = Transform(stft, label_enc)
        >>> transform(ex)
        """
        if (
            self.anchor_shift_sampling_fn is not None
        ):
            assert callable(self.anchor_sampling_fn), type(self.anchor_sampling_fn)
            assert callable(self.anchor_shift_sampling_fn), type(self.anchor_shift_sampling_fn)
            stft = TimeWarpedSTFT(
                base_stft=self.stft,
                anchor_sampling_fn=self.anchor_sampling_fn,
                anchor_shift_sampling_fn=self.anchor_shift_sampling_fn,
            )
        else:
            stft = self.stft

        example = add_label_types(example)
        label_types = example.pop('label_types')
        unlabeled = example.pop('unlabeled')

        example = stft(example)
        seq_len = example['stft'].shape[1]

        weak_labels = [
            (0, 1, self.label_encoder.encode(event_label))
            for event_label in example[self.label_encoder.label_key]
        ]
        weak_targets = self.label_encoder.encode_alignment(
            weak_labels, seq_len=1)[0]

        boundary_labels = {}
        start_frames_key = f'{self.label_encoder.label_key}_start_frames'
        stop_frames_key = f'{self.label_encoder.label_key}_stop_frames'
        for i, event_label in enumerate(example[self.label_encoder.label_key]):
            if label_types[i] not in ['boundaries', 'strong']:
                continue
            if event_label not in boundary_labels:
                boundary_labels[event_label] = (
                    example[start_frames_key][i],
                    example[stop_frames_key][i],
                )
            else:
                boundary_labels[event_label] = (
                    min(boundary_labels[event_label][0], example[start_frames_key][i]),
                    max(boundary_labels[event_label][1], example[stop_frames_key][i]),
                )

        boundary_labels = [
            (
                boundary_labels[event_label][0],
                boundary_labels[event_label][1],
                self.label_encoder.encode(event_label),
            ) for event_label in boundary_labels
        ]
        boundary_targets = self.label_encoder.encode_alignment(
            boundary_labels, seq_len=seq_len)

        strong_labels = [
            (
                example[start_frames_key][i],
                example[stop_frames_key][i],
                self.label_encoder.encode(event_label),
            ) for i, event_label in enumerate(example[self.label_encoder.label_key])
            if label_types[i] == 'strong'
        ]
        strong_targets = self.label_encoder.encode_alignment(
            strong_labels, seq_len=seq_len)

        if unlabeled:
            weak_targets += (1-weak_targets) * 0.5
            boundary_targets += (1-boundary_targets) * 0.5
            strong_targets += (1-strong_targets) * 0.5
        else:
            overall_targets = self.label_encoder(example)['events']
            boundary_targets += (1-boundary_targets) * 0.5 * overall_targets
            strong_targets += (1-strong_targets) * 0.5 * overall_targets

        return {
            'dataset': example['dataset'],
            'example_id': example['example_id'],
            'stft': example['stft'],
            'seq_len': example['stft'].shape[1],
            'weak_targets': weak_targets,
            'boundary_targets': boundary_targets.T,
            'strong_targets': strong_targets.T,
        }
