from pathlib import Path

import numpy as np
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.modules.augment import (
    MelWarping, LogTruncNormalSampler, TruncExponentialSampler
)
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from pb_sed.experiments.dcase_2020_task_4 import data
from pb_sed.models.cnn import CNN
from pb_sed.paths import storage_root
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

ex_name = 'dcase_2020_tag_conditioned_cnn'
ex = Exp(ex_name)


@ex.config
def config():
    debug = False

    # Data configuration
    repetitions = {
        'desed_real_weak_pseudo_strong_2020-07-04-22-16-46': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-05': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-19': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-33': 0,
        'desed_real_weak_pseudo_strong_2020-09-07-14-40-09': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-12-06': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-07-04-22-33-13': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-28-33': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-28-54': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-28-52': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-07-15-09-15': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-29-11': 0,
        'desed_synthetic': 2,
    }
    audio_reader = {
        'source_sample_rate': None,
        'target_sample_rate': 16000,
    }
    cached_datasets = [] if debug else [
        'desed_real_weak_pseudo_strong_2020-07-04-22-16-46',
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-05',
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-19',
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-33',
        'desed_real_weak_pseudo_strong_2020-09-07-14-40-09',
        'desed_real_weak_pseudo_strong_2020-09-06-13-12-06',
        'desed_synthetic',
    ]
    stft = {
        'shift': 320,
        'window_length': 960,
        'size': 1024,
        'fading': None,
        'pad': False,
    }

    mixup_probs = (.5, .5)
    max_mixup_length = int(12.*audio_reader['target_sample_rate']/stft['shift']) + 1
    batch_size = 24
    min_examples = {
        'desed_real_weak_pseudo_strong_2020-07-04-22-16-46': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-05': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-19': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-10-33': 0,
        'desed_real_weak_pseudo_strong_2020-09-07-14-40-09': 0,
        'desed_real_weak_pseudo_strong_2020-09-06-13-12-06': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-07-04-22-33-13': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-28-33': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-28-54': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-28-52': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-07-15-09-15': 0,
        'desed_real_unlabel_in_domain_pseudo_strong_2020-09-06-13-29-11': 0,
        'desed_synthetic': 0,
    }
    num_workers = 8
    prefetch_buffer = 10 * batch_size
    max_total_size = None
    max_padding_rate = 0.05
    bucket_expiration = 2000 * batch_size

    # Trainer configuration
    subdir = str(Path(ex_name) / timeStamped('')[1:])
    trainer = {
        'model': {
            'factory':  CNN,
            'tag_conditioning': True,
            'feature_extractor': {
                'sample_rate': audio_reader['target_sample_rate'],
                'fft_length': stft['size'],
                'n_mels': 128,
                'warping_fn': {
                    'factory': MelWarping,
                    'alpha_sampling_fn': {
                        'factory': LogTruncNormalSampler,
                        'scale': .08,
                        'truncation': np.log(1.3),
                    },
                    'fhi_sampling_fn': {
                        'factory': TruncExponentialSampler,
                        'scale': .5,
                        'truncation': 5.,
                    },
                },
                'max_resample_rate': 1.,
                'blur_sigma': .5,
                'n_time_masks': 0,
                'max_masked_time_steps': 70,
                'max_masked_time_rate': .2,
                'n_mel_masks': 1,
                'max_masked_mel_steps': 20,
                'max_masked_mel_rate': .2,
                'max_noise_scale': .2,
            },
            'cnn_2d': {
                'out_channels': [16, 16, 32, 32, 64, 64, 128, 128, 256],
                'pool_size': [1, (2, 1), 1, (2, 1), 1, (2, 1), 1, (2, 1), (2, 1)],
                'output_layer': False,
                'kernel_size': 3,
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0,
            },
            'cnn_1d': {
                'out_channels': 2*[256] + [10],
                'kernel_size': 3,
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0,
            },
        },
        'optimizer': {
            'factory': Adam,
            'lr': 5e-4,
            'gradient_clipping': 20.,
            'weight_decay': 1e-6,
        },
        'storage_dir': str(storage_root / subdir),
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1000, 'iteration'),
        'stop_trigger': (40000, 'iteration')
    }
    Trainer.get_config(trainer)
    resume = False
    rampup_steps = 1000
    lr_decay_step = 15000

    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def train(
        _run,
        repetitions, audio_reader, cached_datasets, stft,
        mixup_probs, max_mixup_length,
        num_workers, prefetch_buffer,
        batch_size, max_padding_rate, bucket_expiration, min_examples,
        rampup_steps, lr_decay_step,
        trainer, resume,
):

    print_config(_run)
    trainer = Trainer.from_config(trainer)

    train_iter = data.get_train(
        repetitions=repetitions,
        audio_reader=audio_reader, stft=stft,
        mixup_probs=mixup_probs, max_mixup_length=max_mixup_length,
        num_workers=num_workers, prefetch_buffer=prefetch_buffer,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=bucket_expiration, min_examples=min_examples,
        storage_dir=trainer.storage_dir,
        add_alignment=True,
        cached_datasets=cached_datasets,
    )
    validation_set = data.get_dataset(
        'desed_real_validation', audio_reader=audio_reader,
    )
    validation_iter = data.prepare_dataset(
        validation_set,
        storage_dir=trainer.storage_dir,
        audio_reader=audio_reader, stft=stft,
        num_workers=num_workers, prefetch_buffer=prefetch_buffer,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=bucket_expiration,
        add_alignment=True,
    )

    trainer.test_run(train_iter, validation_iter)

    trainer.register_validation_hook(
        validation_iter, metric='mean_fscore', maximize=True
    )
    trainer.register_hook(LRAnnealingHook(
        trigger=(100, 'iteration'),
        breakpoints=[
            (0, 0.),
            (rampup_steps, 1.),
            (lr_decay_step, 1.),
            (lr_decay_step, 1/5),
        ],
        unit='iteration',
    ))

    trainer.train(train_iter, resume=resume)
