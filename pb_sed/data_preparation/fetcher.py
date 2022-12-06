import dataclasses
from padertorch.contrib.je.data.transforms import Collate
from padertorch.contrib.je.data.utils import DynamicExtendedTimeSeriesBucket


@dataclasses.dataclass
class DataFetcher:
    prefetch_workers: int = 8
    global_shuffle: bool = False
    local_shuffle_buffer_size: int = 0
    batch_size: int = None
    max_padding_rate: float = 0.1
    min_label_diversity_in_batch: int = 0  # must not be chosen larger than the number of classes in the input dataset, otherwise fetching gets stuck
    min_dataset_examples_in_batch: dict = None  # the min percentage of a certain dataset in a batch shouldn't exceed the percentage of the dataset in the overall dataset
    bucket_expiration: int = None
    max_bucket_buffer_size: int = None
    drop_incomplete: bool = False

    def __call__(self, dataset, batched_input=False):
        if self.global_shuffle:
            dataset = dataset.shuffle(reshuffle=True)

        if self.prefetch_workers > 0:
            dataset = dataset.prefetch(
                self.prefetch_workers, 2 * self.prefetch_workers
            )

        if batched_input:
            dataset = dataset.unbatch()

        if self.global_shuffle and not batched_input and self.local_shuffle_buffer_size > 0:
            raise AssertionError('using local_shuffle_buffer_size > 0 when global_shuffle is True and batched_input is False has no effect and is therefore inefficient')
        elif self.local_shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                reshuffle=True, buffer_size=self.local_shuffle_buffer_size
            )

        if self.batch_size is not None:
            dataset = dataset.batch_dynamic_bucket(
                bucket_cls=DynamicExtendedTimeSeriesBucket,
                batch_size=self.batch_size,
                max_padding_rate=self.max_padding_rate,
                len_key="seq_len",
                min_label_diversity=self.min_label_diversity_in_batch,
                label_key="weak_targets",
                min_dataset_examples=self.min_dataset_examples_in_batch,
                expiration=self.bucket_expiration,
                max_buffered_examples=self.max_bucket_buffer_size,
                drop_incomplete=self.drop_incomplete,
                sort_key="seq_len", reverse_sort=True,
            ).map(Collate()).prefetch(num_workers=1, buffer_size=4)
        return dataset
