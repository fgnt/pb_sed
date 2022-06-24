
import numpy as np
import torch
import time
from pathlib import Path
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

from paderbox.utils.timer import timeStamped
from paderbox.utils.random_utils import (
    LogTruncatedNormal, TruncatedExponential
)
from paderbox.utils.nested import flatten, deflatten
from paderbox.transform.module_fbank import MelWarping
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.trigger import AllTrigger, EndTrigger, NotTrigger
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from padertorch.contrib.je.modules.transformer import TransformerStack

from pb_sed.models import strong_label
from pb_sed.paths import storage_root
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.experiments.strong_label_crnn.tuning import ex as tuning
from pb_sed.paths import database_jsons_dir


ex_name = 'strong_label_crnn_training'
ex = Exp(ex_name)


@ex.config
def config():
    delay = 0
    debug = False
    timestamp = timeStamped('')[1:] + ('_debug' if debug else '')
    group_name = timestamp
    database_name = 'desed'
    storage_dir = str(storage_root / 'strong_label_crnn' / database_name / 'training' / group_name / timestamp)

    init_ckpt_path = None
    frozen_cnn_2d_layers = 0
    frozen_cnn_1d_layers = 0

    # Data provider
    if database_name == 'desed':
        external_data = True
        batch_size = 32
        data_provider = {
            'factory': DESEDProvider,
            'json_path':
                str(database_jsons_dir / 'desed_pseudo_labeled_with_external.json') if external_data
                else str(database_jsons_dir / 'desed_pseudo_labeled_without_external.json'),
            'train_set': {
                'train_weak': 10 if external_data else 20,
                'train_strong': 10 if external_data else 0,
                'train_synthetic20': 2,
                'train_synthetic21': 1,
                'train_unlabel_in_domain': 2,
            },
            'cached_datasets': None if debug else ['train_weak', 'train_synthetic20'],
            'train_fetcher': {
                'batch_size': batch_size,
                'prefetch_workers': batch_size,
                'min_dataset_examples_in_batch': {
                    'train_weak': int(3*batch_size/32),
                    'train_strong': int(6*batch_size/32) if external_data else 0,
                    'train_synthetic20': int(1*batch_size/32),
                    'train_synthetic21': int(2*batch_size/32),
                    'train_unlabel_in_domain': 0,
                },
            },
            'storage_dir': storage_dir,
        }
        num_events = 10
        DESEDProvider.get_config(data_provider)

        validation_set_name = 'validation'
        validation_ground_truth_filepath = None
        weak_label_crnn_hyper_params_dir = ''
        eval_set_name = 'eval_public'
        eval_ground_truth_filepath = None

        num_iterations = 45000 if init_ckpt_path is None else 20000
        checkpoint_interval = 1000
        summary_interval = 100
        back_off_patience = None
        lr_decay_step = 30000 if back_off_patience is None else None
        lr_decay_factor = 1/5
        lr_rampup_steps = 1000 if init_ckpt_path is None else None
        gradient_clipping = 1e10 if init_ckpt_path is None else 1
    else:
        raise ValueError(f'Unknown database {database_name}.')

    # Trainer configuration
    net_config = 'shallow'
    if net_config == 'shallow':
        m = 1
        cnn = {
            'cnn_2d': {
                'out_channels': [
                    16*m, 16*m, 32*m, 32*m, 64*m, 64*m, 128*m, 128*m, min(256*m, 512),
                ],
                'pool_size': 4*[1, (2, 1)] + [1],
                'kernel_size': 3,
                'norm': 'batch',
                'norm_kwargs': {'eps': 1e-3},
                'activation_fn': 'relu',
                'dropout': .0,
                'output_layer': False,
            },
            'cnn_1d': {
                'out_channels': 3*[256*m],
                'kernel_size': 3,
                'norm': 'batch',
                'norm_kwargs': {'eps': 1e-3},
                'activation_fn': 'relu',
                'dropout': .0,
                'output_layer': False,
            },
        }
    elif net_config == 'deep':
        m = 2
        cnn = {
            'cnn_2d': {
                'out_channels': (
                    4*[16*m] + 4*[32*m] + 4*[64*m] + 4*[128*m] + [256*m, min(256*m, 512)]
                ),
                'pool_size': 4*[1, 1, 1, (2, 1)] + [1, 1],
                'kernel_size': 9*[3, 1],
                'residual_connections': [
                    None, None, 4, None,
                    6, None, 8, None,
                    10, None, 12, None,
                    14, None, 16, None,
                    None, None
                ],
                'norm': 'batch',
                'norm_kwargs': {'eps': 1e-3},
                'activation_fn': 'relu',
                'pre_activation': True,
                'dropout': .0,
                'output_layer': False,
            },
            'cnn_1d': {
                'out_channels': 8*[256*m],
                'kernel_size': [1] + 3*[3, 1] + [1],
                'residual_connections': [None, 3, None, 5, None, 7, None, None],
                'norm': 'batch',
                'norm_kwargs': {'eps': 1e-3},
                'activation_fn': 'relu',
                'pre_activation': True,
                'dropout': .0,
                'output_layer': False,
            },
        }
    else:
        raise ValueError(f'Unknown net_config {net_config}')

    if init_ckpt_path is not None:
        cnn['conditional_dims'] = 0

    trainer = {
        'model': {
            'factory': strong_label.CRNN,
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
            'cnn': cnn,
            'rnn': {
                'hidden_size': 256*m,
                'num_layers': 2,
                'dropout': .0,
                'output_net': {
                    'out_channels': [
                        256*m,
                        num_events
                    ],
                    'kernel_size': 1,
                    'norm': 'batch',
                    'activation_fn': 'relu',
                    'dropout': .0,
                }
            },
            'labelwise_metrics': ('fscore_strong',),
        },
        'optimizer': {
            'factory': Adam,
            'lr': 5e-4,
            'gradient_clipping': gradient_clipping,
            # 'weight_decay': 1e-6,
        },
        'summary_trigger': (summary_interval, 'iteration'),
        'checkpoint_trigger': (checkpoint_interval, 'iteration'),
        'stop_trigger': (num_iterations, 'iteration'),
        'storage_dir': storage_dir,
    }
    del cnn
    use_transformer = False
    if use_transformer:
        trainer['model']['rnn']['factory'] = TransformerStack
        trainer['model']['rnn']['hidden_size'] = 320
        trainer['model']['rnn']['num_heads'] = 10
        trainer['model']['rnn']['num_layers'] = 3
        trainer['model']['rnn']['dropout'] = 0.1
    Trainer.get_config(trainer)

    resume = False
    assert resume or not Path(trainer['storage_dir']).exists()
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def train(
        _run, debug,
        data_provider, trainer,
        lr_rampup_steps, back_off_patience, lr_decay_step, lr_decay_factor,
        init_ckpt_path, frozen_cnn_2d_layers, frozen_cnn_1d_layers,
        resume, delay,
        validation_set_name, validation_ground_truth_filepath,
        weak_label_crnn_hyper_params_dir,
        eval_set_name, eval_ground_truth_filepath,
):
    print()
    print('##### Training #####')
    print()
    print_config(_run)
    assert (back_off_patience is None) or (lr_decay_step is None), (back_off_patience, lr_decay_step)
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
    trainer.model.label_mapping = []
    for idx, label in sorted(data_provider.train_transform.label_encoder.inverse_label_mapping.items()):
        assert idx == len(trainer.model.label_mapping), (idx, label, len(trainer.model.label_mapping))
        trainer.model.label_mapping.append(label.replace(', ', '__').replace(' ', '').replace('(', '_').replace(')', '_').replace("'", ''))
    print('Params', sum(p.numel() for p in trainer.model.parameters()))

    if init_ckpt_path is not None:
        print('Load init params')
        state_dict = deflatten(torch.load(init_ckpt_path, map_location='cpu')['model'], maxdepth=1)
        trainer.model.cnn.load_state_dict(state_dict['cnn'])
    if frozen_cnn_2d_layers:
        print(f'Freeze {frozen_cnn_2d_layers} cnn_2d layers')
        trainer.model.cnn.cnn_2d.freeze(frozen_cnn_2d_layers)
    if frozen_cnn_1d_layers:
        print(f'Freeze {frozen_cnn_1d_layers} cnn_1d layers')
        trainer.model.cnn.cnn_1d.freeze(frozen_cnn_1d_layers)

    def add_tag_condition(example):
        example["tag_condition"] = example["weak_targets"]
        return example

    train_set = data_provider.get_train_set().map(add_tag_condition)
    validate_set = data_provider.get_validate_set().map(add_tag_condition)

    if validate_set is not None:
        trainer.test_run(train_set, validate_set)
        trainer.register_validation_hook(
            validate_set, metric='macro_fscore_strong', maximize=True,
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

    if validation_set_name:
        tuning.run(
            config_updates={
                'debug': debug,
                'weak_label_crnn_hyper_params_dir': weak_label_crnn_hyper_params_dir,
                'strong_label_crnn_dirs': [str(trainer.storage_dir)],
                'validation_set_name': validation_set_name,
                'validation_ground_truth_filepath': validation_ground_truth_filepath,
                'eval_set_name': eval_set_name,
                'eval_ground_truth_filepath': eval_ground_truth_filepath,
            }
        )
