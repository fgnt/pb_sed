
import numpy as np
import numbers
from lazy_dataset import Dataset
from pb_sed.data_preparation.utils import add_label_types


class MixtureDataset(Dataset):
    """
    >>> ds = MixtureDataset(range(10), range(10), 2, (lambda x: x))
    >>> list(ds)
    >>> ds[2]
    """
    def __init__(self, input_dataset, mixin_dataset, mix_interval, mix_fn):
        """
        Mixes examples from input_dataset and mixin_dataset.

        Args:
            input_dataset: lazy dataset providing example dict with key audio_length.
            mixin_dataset:
            mix_interval:
            mix_fn:
        """
        assert len(mixin_dataset) >= len(input_dataset), (len(mixin_dataset), len(input_dataset))
        self.input_dataset = input_dataset
        self.mixin_dataset = mixin_dataset
        assert mix_interval >= 1
        self.mix_interval = mix_interval
        self.mix_fn = mix_fn

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            example = self.input_dataset[item]
            if (item % self.mix_interval) < 1:
                mixin_item = int(item // self.mix_interval)
                mixin_example = self.mixin_dataset[mixin_item]
                return self.mix_fn([example, mixin_example])
            else:
                return example
        else:
            return super().__getitem__(item)

    def __iter__(self):
        mixin_iter = iter(self.mixin_dataset)
        for i, example in enumerate(self.input_dataset):
            if (i % self.mix_interval) < 1:
                mixin_example = next(mixin_iter)
                yield self.mix_fn([example, mixin_example])
            else:
                yield example

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            mixin_dataset=self.mixin_dataset.copy(freeze=freeze),
            mix_interval=self.mix_interval,
            mix_fn=self.mix_fn,
        )

    @property
    def indexable(self):
        return self.input_dataset.indexable


class SuperposeEvents:
    """
    >>> mix_fn = SuperposeEvents(min_overlap=0.5)
    >>> example1 = {'example_id': '0', 'dataset': '0', 'audio_data': np.ones((1, 8)), 'events': ['a'], 'events_start_samples': [2], 'events_stop_samples': [8], 'label_types': ['strong'],}
    >>> example2 = {'example_id': '1', 'dataset': '1', 'audio_data': -np.ones((1, 10)), 'events': ['a', 'b'], 'events_start_samples': [0, 1], 'events_stop_samples': [8, 4], 'label_types': ['weak', 'strong']}
    >>> overlap = np.array([(mix_fn([example1, example2])['audio_data'] == 0).sum() for _ in range(10000)])
    >>> [(overlap == i).sum() for i in range(10)]
    """
    def __init__(
            self, min_overlap=1., max_length_in_samples=None, fade_length=0,
            label_key='events'
    ):
        self.min_overlap = min_overlap
        self.max_length_in_samples = max_length_in_samples
        self.fade_length = fade_length
        self.label_key = label_key

    def __call__(self, components):
        assert len(components) > 0
        components = [add_label_types(comp) for comp in components]
        start_indices = [0]
        stop_indices = [components[0]['audio_data'].shape[1]]
        for comp in components[1:]:
            seq_len = comp['audio_data'].shape[1]
            seq_len_min = min(seq_len, components[0]['audio_data'].shape[1])
            min_overlap = int(np.ceil(seq_len_min * self.min_overlap))
            min_start = -(seq_len - min_overlap)
            max_start = components[0]['audio_data'].shape[1] - min_overlap
            if self.max_length_in_samples is not None:
                assert seq_len <= self.max_length_in_samples, (seq_len, self.max_length_in_samples)
                min_start = max(
                    min_start, max(stop_indices) - self.max_length_in_samples
                )
                max_start = min(
                    max_start, min(start_indices) + self.max_length_in_samples - seq_len
                )
            start_indices.append(
                int(np.floor(min_start + np.random.rand() * (max_start - min_start + 1)))
            )
            stop_indices.append(start_indices[-1] + seq_len)
        start_indices = np.array(start_indices)
        stop_indices = np.array(stop_indices)
        stop_indices -= start_indices.min()
        start_indices -= start_indices.min()

        audio_shape = list(components[0]['audio_data'].shape)
        audio_shape[1] = stop_indices.max()
        mixed_audio = np.zeros(audio_shape, dtype=components[0]['audio_data'].dtype)
        events = []
        label_types = []
        events_start_samples = []
        events_stop_samples = []
        for comp, start, stop in zip(components, start_indices, stop_indices):
            mixin_audio = np.copy(comp['audio_data'])
            if self.fade_length > 0:
                assert mixin_audio.shape[1] > 2 * self.fade_length, mixin_audio.shape
                raised_cos = 1/2+np.cos(np.pi*np.arange(1, self.fade_length+1) / (self.fade_length+1))/2
                if start > 0:
                    mixin_audio[:, :self.fade_length] *= raised_cos[::-1]
                if stop < audio_shape[1]:
                    assert mixin_audio.shape[1] > self.fade_length, mixin_audio.shape
                    mixin_audio[:, -self.fade_length:] *= raised_cos
            mixed_audio[:, start:stop] += mixin_audio
            events.extend(comp[self.label_key])
            label_types.extend(comp['label_types'])
            assert f'{self.label_key}_start_samples' in comp, comp.keys()
            assert f'{self.label_key}_stop_samples' in comp, comp.keys()
            events_start_samples.extend([
                event_start + start
                for event_start in comp[f'{self.label_key}_start_samples']
            ])
            events_stop_samples.extend([
                event_stop + start
                for event_stop in comp[f'{self.label_key}_stop_samples']
            ])
        # mixed_audio /= np.sqrt(len(components))

        mix = {
            'example_id': '+'.join([comp['example_id'] for comp in components]),
            'dataset': '+'.join(sorted(set([comp['dataset'] for comp in components]))),
            'audio_data': mixed_audio,
            'seq_len': mixed_audio.shape[1],
            self.label_key: events,
            f'{self.label_key}_start_samples': events_start_samples,
            f'{self.label_key}_stop_samples': events_stop_samples,
            'label_types': label_types,
            'unlabeled': any([comp['unlabeled'] for comp in components]),
        }
        return mix
