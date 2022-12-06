
import dataclasses
import lazy_dataset
from typing import Mapping, Sequence
from padertorch.configurable import _DogmaticConfig
from pb_sed.data_preparation.provider import DataProvider
from pb_sed.paths import database_jsons_dir


@dataclasses.dataclass
class AudioSetProvider(DataProvider):
    add_ancestor_events: bool = False

    def get_raw(
            self, dataset_names_or_raw_datasets,
            discard_labelless_examples=False,
            filter_example_ids=None,
    ):
        raw_dataset = super().get_raw(
            dataset_names_or_raw_datasets=dataset_names_or_raw_datasets,
            discard_labelless_examples=discard_labelless_examples,
            filter_example_ids=filter_example_ids,
        )
        if self.add_ancestor_events and isinstance(raw_dataset, lazy_dataset.Dataset):
            ontology = self.db.data['ontology']
            ds_names = self._get_dataset_names(self.train_set, self.validate_set)
            event_classes = set(self.db.data['strong_event_classes']) \
                if self.strongly_labeled_data(ds_names) \
                else set(self.db.data['strong_event_classes'])

            def add_ancestor_events(example):
                for idx, event in enumerate(example['events']):
                    if event not in event_classes:
                        continue
                    for ancestor in ontology[event]['ancestor_names']:
                        if ancestor not in event_classes:
                            continue
                        example['events'].append(ancestor)
                        if 'events_start_times' in example:
                            example['events_start_times'].append(example['events_start_times'][idx])
                            example['events_stop_times'].append(example['events_stop_times'][idx])
                        if 'label_types' in example:
                            example['label_types'].append(example['label_types'][idx])

                if 'events_start_times' in example:
                    sort_idx = sorted(range(len(example['events'])), key=lambda i: example['events_start_times'][i])
                    for key in ['events', 'events_start_times', 'events_stop_times', 'label_types']:
                        if key in example:
                            example[key] = [example[key][i] for i in sort_idx]
                return example

            raw_dataset = raw_dataset.map(add_ancestor_events)
        return raw_dataset

    @classmethod
    def _get_dataset_names(cls, train_set, validate_set):
        dataset_names = []
        for ds in [train_set, validate_set]:
            if isinstance(ds, str):
                dataset_names.append(ds)
            elif isinstance(ds, (Mapping, _DogmaticConfig)):
                dataset_names.extend(list(ds.keys()))
            elif isinstance(ds, Sequence):
                dataset_names.extend(list(ds))
            elif ds is not None:
                raise ValueError(type(ds))
        assert len(dataset_names) > 0, dataset_names
        return dataset_names

    @classmethod
    def strongly_labeled_data(cls, dataset_names):
        if (
                'balanced_train' in dataset_names
                or 'unbalanced_train' in dataset_names
                or 'eval' in dataset_names
        ):
            assert 'train_strong' not in dataset_names
            assert 'eval_strong' not in dataset_names
            return False
        else:
            return True

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['json_path'] = str(database_jsons_dir / 'audioset.json')
        config['validate_set'] = 'eval'
        super().finalize_dogmatic_config(config)
        ds_names = cls._get_dataset_names(
            config['train_set'], config['validate_set'])
        if cls.strongly_labeled_data(ds_names):
            num_events = 456
        else:
            num_events = 527
        config['train_fetcher']['min_label_diversity_in_batch'] = min(
            num_events, config['train_fetcher']['batch_size']
        )
