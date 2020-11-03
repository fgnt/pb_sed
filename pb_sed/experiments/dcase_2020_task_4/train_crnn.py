"""
This script trains a (forward-backward) CRNN model.

For model details see
http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Ebbers_69.pdf

Available pseudo labeled unlabel_in_domain datasets:

weak pseudo labels from FBCRNN ensembles trained on weak+synthetic:
unlabel_in_domain_pseudo_weak_2020-07-03-20-48-45
unlabel_in_domain_pseudo_weak_2020-07-03-20-49-48
unlabel_in_domain_pseudo_weak_2020-07-03-20-52-19
unlabel_in_domain_pseudo_weak_2020-07-03-21-00-48
unlabel_in_domain_pseudo_weak_2020-07-03-21-05-34

weak pseudo labels from FBCRNN ensembles trained on weak+synthetic+unlabel_in_domain_pseudo_weak:
unlabel_in_domain_pseudo_weak_2020-07-04-13-10-05
unlabel_in_domain_pseudo_weak_2020-07-04-13-10-19
unlabel_in_domain_pseudo_weak_2020-07-04-13-10-33
unlabel_in_domain_pseudo_weak_2020-07-04-13-11-09
unlabel_in_domain_pseudo_weak_2020-07-04-13-12-06

Example calls:
train FBCRNN on weak+synthetic:
python -m pb_sed.experiments.dcase_2020_task_4.train_crnn

train CRNN (w/o backward tagging) on weak+synthetic w/o framewise loss:
python -m pb_sed.experiments.dcase_2020_task_4.train_crnn with trainer.model.rnn_bwd=None trainer.model.clf_bwd=None trainer.model.framewise_training=False

train FBCRNN on weak+synthetic+unlabel_in_domain_pseudo_weak:
python -m pb_sed.experiments.dcase_2020_task_4.train_crnn with 'unlabel_in_domain_pseudo_weak_timestamp=2020-07-04-13-10-05'
"""
from pathlib import Path

import numpy as np
import torch
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.modules.augment import (
    MelWarping, LogTruncNormalSampler, TruncExponentialSampler
)
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from pb_sed.experiments.dcase_2020_task_4 import data
from pb_sed.models.crnn import CRNN
from pb_sed.paths import storage_root
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

ex_name = 'dcase_2020_crnn'
ex = Exp(ex_name)


@ex.config
def config():
    debug = False

    # Data configuration
    unlabel_in_domain_pseudo_weak_timestamp = None
    dataset_repetitions = {
        'weak': 10,
        'synthetic': 2,
    }
    if unlabel_in_domain_pseudo_weak_timestamp is not None:
        dataset_repetitions[f'unlabel_in_domain_pseudo_weak_{unlabel_in_domain_pseudo_weak_timestamp}'] = 1
    audio_reader = {
        'source_sample_rate': None,
        'target_sample_rate': 16000,
    }
    cached_datasets = [] if debug else ['weak', 'synthetic']
    stft = {
        'shift': 320,
        'window_length': 960,
        'size': 1024,
        'fading': None,
        'pad': False,
    }

    mixup_probs = (1/3, 2/3)
    max_mixup_length = int(15.*audio_reader['target_sample_rate']/stft['shift'])+1
    batch_size = 16
    min_examples = {
        **{ds: 0 for ds in dataset_repetitions},
        'weak': int(batch_size/3),
    }
    if unlabel_in_domain_pseudo_weak_timestamp is not None:
        min_examples[f'unlabel_in_domain_pseudo_weak_{unlabel_in_domain_pseudo_weak_timestamp}'] = 0
    num_workers = 8
    prefetch_buffer = 10 * batch_size
    max_total_size = None
    max_padding_rate = 0.05
    bucket_expiration = 2000 * batch_size

    # Trainer configuration
    subdir = str(Path(ex_name) / timeStamped('')[1:])
    trainer = {
        'model': {
            'factory':  CRNN,
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
                'n_time_masks': 1,
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
                'kernel_size': [3, (3, 1), 3, (3, 1), 3, (3, 1), 3, (3, 1), (3, 1)],
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0,
            },
            'cnn_1d': {
                'out_channels': 3*[256],
                'output_layer': False,
                'kernel_size': 3,
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0,
            },
            'rnn_fwd': {
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': .0,
            },
            'clf_fwd': {
                'out_channels': [256, 10],
                'kernel_size': 1,
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
    cnn_2d_init = None
    freeze_cnn_2d = False
    cnn_1d_init = None
    freeze_cnn_1d = False
    Trainer.get_config(trainer)
    resume = False
    rampup_steps = 1000
    lr_decay_step = 15000

    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def train(
        _run,
        dataset_repetitions, audio_reader, cached_datasets, stft,
        mixup_probs, max_mixup_length,
        num_workers, prefetch_buffer,
        batch_size, max_padding_rate, bucket_expiration, min_examples,
        rampup_steps, lr_decay_step,
        trainer, cnn_2d_init, freeze_cnn_2d, cnn_1d_init, freeze_cnn_1d,
        resume,
):

    print_config(_run)
    trainer = Trainer.from_config(trainer)
    if cnn_2d_init is not None:
        print('Load cnn_2d params')
        state_dict = torch.load(cnn_2d_init, map_location='cpu')
        trainer.model._cnn_2d.load_state_dict(state_dict)
    if freeze_cnn_2d:
        print('Freeze cnn_2d params')
        for param in trainer.model._cnn_2d.parameters():
            param.requires_grad = False
    if cnn_1d_init is not None:
        print('Load cnn_1d params')
        state_dict = torch.load(cnn_1d_init, map_location='cpu')
        trainer.model._cnn_1d.load_state_dict(state_dict)
    if freeze_cnn_1d:
        print('Freeze cnn_1d params')
        for param in trainer.model._cnn_1d.parameters():
            param.requires_grad = False

    train_iter = data.get_train(
        dataset_repetitions=dataset_repetitions,
        audio_reader=audio_reader, stft=stft,
        mixup_probs=mixup_probs, max_mixup_length=max_mixup_length,
        num_workers=num_workers, prefetch_buffer=prefetch_buffer,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=bucket_expiration, min_examples=min_examples,
        storage_dir=trainer.storage_dir,
        cached_datasets=cached_datasets,
    )
    validation_set = data.get_dataset(
        'validation', audio_reader=audio_reader,
    )
    validation_iter = data.prepare_dataset(
        validation_set,
        storage_dir=trainer.storage_dir,
        audio_reader=audio_reader, stft=stft,
        num_workers=num_workers, prefetch_buffer=prefetch_buffer,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=bucket_expiration,
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
