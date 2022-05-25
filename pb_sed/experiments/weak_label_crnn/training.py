
import numpy as np
import time
from pathlib import Path
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

from paderbox.utils.timer import timeStamped
from paderbox.utils.random_utils import (
    LogTruncatedNormal, TruncatedExponential
)
from paderbox.transform.module_fbank import MelWarping
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.trigger import AllTrigger, EndTrigger, NotTrigger
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from padertorch.contrib.je.modules.transformer import TransformerStack

from pb_sed.models import weak_label
from pb_sed.paths import storage_root
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.experiments.weak_label_crnn.tuning import ex as tuning


ex_name = 'weak_label_crnn_training'
ex = Exp(ex_name)


@ex.config
def config():
    delay = 0
    debug = False
    timestamp = timeStamped('')[1:] + ('_debug' if debug else '')
    group_name = timestamp
    storage_dir = str(storage_root / ex_name / group_name / timestamp)

    # Data provider
    batch_size = 16
    data_provider = {
        'factory': DESEDProvider,
        'train_set': {
            'train_weak': 20,
            'train_synthetic20': 2,
            'train_synthetic21': 1,
            'train_unlabel_in_domain': 0,
        },
        'cached_datasets': [] if debug else ['train_weak', 'train_synthetic20'],
        'train_fetcher': {
            'batch_size': batch_size,
            'min_dataset_examples_in_batch': {
                'train_weak': int(6*batch_size/16),
                'train_synthetic20': int(1*batch_size/16),
                'train_synthetic21': int(2*batch_size/16),
                'train_unlabel_in_domain': 0,
            },
        },
        'storage_dir': storage_dir,
    }
    num_events = 10
    DESEDProvider.get_config(data_provider)

    validation_set_name = 'validation'
    validation_ground_truth_filepath = None
    eval_set_name = 'eval_public'
    eval_ground_truth_filepath = None

    num_iterations = 30000 + 15000*(data_provider['train_set']['train_unlabel_in_domain'] > 0)
    lr_decay_step = 20000 + 10000*(data_provider['train_set']['train_unlabel_in_domain'] > 0)
    lr_decay_factor = 1/5
    lr_rampup_steps = 1000

    # Trainer configuration
    k = 1
    trainer = {
        'model': {
            'factory': weak_label.CRNN,
            'feature_extractor': {
                'sample_rate':
                    data_provider['audio_reader']['target_sample_rate'],
                'stft_size': data_provider['train_transform']['stft']['size'],
                'number_of_filters': 128,
                'frequency_warping_fn': {
                    'factory': MelWarping,
                    'warp_factor_sampling_fn': {
                        'factory': LogTruncatedNormal,
                        'scale': .08,
                        'truncation': np.log(1.3),
                    },
                    'boundary_frequency_ratio_sampling_fn': {
                        'factory': TruncatedExponential,
                        'scale': .5,
                        'truncation': 5.,
                    },
                    'highest_frequency': data_provider['audio_reader']['target_sample_rate']/2
                },
                # 'blur_sigma': .5,
                'n_time_masks': 1,
                'max_masked_time_steps': 70,
                'max_masked_time_rate': .2,
                'n_frequency_masks': 1,
                'max_masked_frequency_bands': 20,
                'max_masked_frequency_rate': .2,
                'max_noise_scale': .2,
            },
            'cnn': {
                'cnn_2d': {
                    'out_channels':
                        [16, 16, 32*k, 32*k, 64*k, 64*k, 128*k, 128*k, 256*k],
                    'pool_size': [1, (2, 1), 1, (2, 1), 1, (2, 1), 1, (2, 1), 1],
                    'kernel_size': 3,
                    'norm': 'batch',
                    'activation_fn': 'relu',
                    'dropout': .0,
                    'output_layer': False,
                },
                'cnn_1d': {
                    'out_channels': 3*[256*k],
                    'kernel_size': 3,
                    'norm': 'batch',
                    'activation_fn': 'relu',
                    'dropout': .0,
                    'output_layer': False,
                },
            },
            'rnn_fwd': {
                'hidden_size': 256*k,
                'num_layers': 2,
                'dropout': .0,
                'output_net': {
                    'out_channels': [
                        256*k,
                        num_events
                    ],
                    'kernel_size': 1,
                    'norm': 'batch',
                    'activation_fn': 'relu',
                    'dropout': .0,
                }
            },
            'labelwise_metrics': ('fscore_weak',),
            'strong_fwd_bwd_loss_weight': 1.,
        },
        'optimizer': {
            'factory': Adam,
            'lr': 5e-4,
            # 'gradient_clipping': .7,
            # 'weight_decay': 1e-6,
        },
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1000, 'iteration'),
        'stop_trigger': (num_iterations, 'iteration'),
        'storage_dir': storage_dir,
    }
    use_transformer = False
    if use_transformer:
        trainer['model']['rnn_fwd']['factory'] = TransformerStack
        trainer['model']['rnn_fwd']['hidden_size'] = 320
        trainer['model']['rnn_fwd']['num_heads'] = 10
        trainer['model']['rnn_fwd']['num_layers'] = 3
        trainer['model']['rnn_fwd']['dropout'] = 0.1

    Trainer.get_config(trainer)
    resume = False

    assert resume or not Path(trainer['storage_dir']).exists()
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def train(
        _run, debug,
        data_provider, trainer,
        lr_rampup_steps, lr_decay_step, lr_decay_factor,
        resume, delay,
        validation_set_name, validation_ground_truth_filepath,
        eval_set_name, eval_ground_truth_filepath,
):
    print()
    print('##### Training #####')
    print()
    print_config(_run)
    if delay > 0:
        print(f'Sleep for {delay} seconds.')
        time.sleep(delay)

    data_provider = DESEDProvider.from_config(data_provider)
    data_provider.train_transform.label_encoder.initialize_labels(
        dataset=data_provider.db.get_dataset(data_provider.validate_set),
        verbose=True
    )
    data_provider.test_transform.label_encoder.initialize_labels()
    trainer = Trainer.from_config(trainer)
    print('Params', sum(p.numel() for p in trainer.model.parameters()))

    train_set = data_provider.get_train_set()
    validate_set = data_provider.get_validate_set()

    if validate_set is not None:
        trainer.test_run(train_set, validate_set)
        trainer.register_validation_hook(
            validate_set, metric='macro_fscore_weak', maximize=True,
        )

    breakpoints = []
    if lr_rampup_steps is not None:
        breakpoints += [(0, 0.), (lr_rampup_steps, 1.)]
    if lr_decay_step is not None:
        breakpoints += [(lr_decay_step, 1.), (lr_decay_step, lr_decay_factor)]
    if len(breakpoints) > 0:
        if isinstance(trainer.optimizer, dict):
            names = sorted(trainer.optimizer.keys())
        else:
            names = [None]
        for name in names:
            trainer.register_hook(LRAnnealingHook(
                trigger=AllTrigger(
                    (100, 'iteration'),
                    NotTrigger(EndTrigger(breakpoints[-1][0]+100, 'iteration')),
                ),
                breakpoints=breakpoints,
                unit='iteration',
                name=name,
            ))
    trainer.train(train_set, resume=resume)

    if validation_set_name is not None:
        tuning.run(
            config_updates={
                'debug': debug,
                'crnn_dirs': [str(trainer.storage_dir)],
                'validation_set_name': validation_set_name,
                'validation_ground_truth_filepath': validation_ground_truth_filepath,
                'eval_set_name': eval_set_name,
                'eval_ground_truth_filepath': eval_ground_truth_filepath,
                'data_provider': {
                    'test_fetcher': {
                        'batch_size': data_provider.train_fetcher.batch_size,
                    }
                },
            }
        )
