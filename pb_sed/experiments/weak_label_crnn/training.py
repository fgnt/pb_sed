
import numpy as np
import json
import psutil
import time
import datetime
import torch
from pathlib import Path
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

from paderbox.utils.random_utils import (
    LogTruncatedNormal, TruncatedExponential
)
from paderbox.transform.module_fbank import MelWarping
from paderbox.utils.nested import flatten, deflatten
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.trigger import AllTrigger, EndTrigger, NotTrigger
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from padertorch.contrib.je.modules.rnn import TransformerEncoder

from pb_sed.models import weak_label
from pb_sed.paths import storage_root, database_jsons_dir
from pb_sed.data_preparation.provider import DataProvider
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.database.audioset.provider import AudioSetProvider
from pb_sed.experiments.weak_label_crnn.tuning import ex as tuning


ex_name = 'weak_label_crnn_training'
ex = Exp(ex_name)


@ex.config
def config():
    delay = 0
    debug = False
    dt = datetime.datetime.now()
    timestamp = dt.strftime('%Y-%m-%d-%H-%M-%S-{:02d}').format(int(dt.microsecond/10000)) + ('_debug' if debug else '')
    del dt
    group_name = timestamp
    database_name = 'desed'
    storage_dir = str(storage_root / 'weak_label_crnn' / database_name / 'training' / group_name / timestamp)
    resume = False
    if resume:
        assert Path(storage_dir).exists()
    else:
        assert not Path(storage_dir).exists()
        Path(storage_dir).mkdir(parents=True)

    init_ckpt_path = None
    frozen_cnn_2d_layers = 0
    frozen_cnn_1d_layers = 0
    freeze_norm_stats = True
    finetune_mode = init_ckpt_path is not None

    # Data provider
    if database_name == 'desed':
        external_data = True
        batch_size = 32
        data_provider = {
            'factory': DESEDProvider,
            'train_set': {
                'train_weak': 10 if external_data else 20,
                'train_strong': 10 if external_data else 0,
                'train_synthetic20': 2,
                'train_synthetic21': 1,
                'train_unlabel_in_domain': 0,
            },
            'cached_datasets': None if debug else ['train_weak', 'train_synthetic20'],
            'train_fetcher': {
                'batch_size': batch_size,
                'prefetch_workers': len(psutil.Process().cpu_affinity())-2,
                'min_dataset_examples_in_batch': {
                    'train_weak': int(3*batch_size/32),
                    'train_strong': int(6*batch_size/32) if external_data else 0,
                    'train_synthetic20': int(1*batch_size/32),
                    'train_synthetic21': int(2*batch_size/32),
                    'train_unlabel_in_domain': 0,
                },
            },
            'train_transform': {
                'provide_boundary_targets': True,
            },
            'storage_dir': storage_dir,
        }
        num_events = 10
        DESEDProvider.get_config(data_provider)

        validation_set_name = 'validation'
        validation_ground_truth_filepath = None
        eval_set_name = 'eval_public'
        eval_ground_truth_filepath = None

        num_iterations = int(
            40000 * (1+0.5*(data_provider['train_set']['train_unlabel_in_domain'] > 0)) * 16/batch_size
        )
        checkpoint_interval = int(2000 * 16/batch_size)
        summary_interval = 100
        lr = 5e-4
        n_back_off = 0
        back_off_patience = 10
        lr_decay_steps = [
            int(20000 * (1+0.5*(data_provider['train_set']['train_unlabel_in_domain'] > 0)) * 16/batch_size)
        ] if n_back_off == 0 else []
        lr_decay_factor = 1/5
        lr_rampup_steps = None if finetune_mode else int(2000 * 16/batch_size)
        gradient_clipping = 1 if finetune_mode else 1e10
        strong_fwd_bwd_loss_weight = 1.
        early_stopping_patience = None
    elif database_name == 'audioset':
        batch_size = 32
        data_provider = {
            'factory': AudioSetProvider,
            'train_set': {
                'balanced_train': 1,
                'unbalanced_train': 1,
            },
            'train_fetcher': {
                'batch_size': batch_size,
                'prefetch_workers': len(psutil.Process().cpu_affinity())-2,
            },
            'min_class_examples_per_epoch': 0.01,
            'storage_dir': storage_dir,
        }
        num_events = 527
        AudioSetProvider.get_config(data_provider)

        validation_set_name = None
        validation_ground_truth_filepath = None
        eval_set_name = None
        eval_ground_truth_filepath = None

        num_iterations = int(1000000 * 16/batch_size)
        checkpoint_interval = int(10000 * 16/batch_size)
        summary_interval = int(1000 * 16/batch_size)
        lr = 1e-4
        n_back_off = 0
        back_off_patience = 10
        lr_decay_steps = [
            int(600000 * 16/batch_size),
            int(800000 * 16/batch_size)
        ] if n_back_off == 0 else []
        lr_decay_factor = float(np.sqrt(.1))
        lr_rampup_steps = int(2000 * 16/batch_size)
        early_stopping_patience = None

        gradient_clipping = .1
        strong_fwd_bwd_loss_weight = 0.
    else:
        raise ValueError(f'Unknown database {database_name}.')
    filter_desed_test_clips = False
    hyper_params_tuning_batch_size = batch_size // 2

    # Trainer configuration
    net_config = 'shallow'
    if net_config == 'shallow':
        width = 1
        kernel_size_2d = 3
        out_channels_2d = [
            16*width, 16*width, 32*width, 32*width, 64*width, 64*width,
            128*width, 128*width, min(256*width, 512),
        ]
        residual_connections_2d = None
        pool_sizes_2d = 4*[1, (2, 1)] + [1]
        kernel_size_1d = [1] + 3*[3] + [1]
        residual_connections_1d = None
    elif net_config == 'deep':
        width = 2
        kernel_size_2d = 9*[3, 1]
        out_channels_2d = (
                4*[16*width] + 4*[32*width] + 4*[64*width] + 4*[128*width]
                + [256*width, min(256*width, 512)]
        )
        residual_connections_2d = [
            None, None, 4, None, 6, None, 8, None, 10, None, 12, None,
            14, None, 16, None, None, None
        ]
        pool_sizes_2d = 4*[1, 1, 1, (2, 1)] + [1, 1]
        kernel_size_1d = [1] + 3*[3, 1] + [1]
        residual_connections_1d = [None, 3, None, 5, None, 7, None, None]
    else:
        raise ValueError(f'Unknown net_config {net_config}')

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
                    'highest_frequency': data_provider['audio_reader']['target_sample_rate']/2,
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
                    'out_channels': out_channels_2d,
                    'pool_size': pool_sizes_2d,
                    'kernel_size': kernel_size_2d,
                    'residual_connections': residual_connections_2d,
                    'norm': 'batch',
                    'norm_kwargs': {'eps': 1e-3},
                    'activation_fn': 'relu',
                    'pre_activation': True,
                    'dropout': .0,
                    'output_layer': False,
                },
                'cnn_1d': {
                    'out_channels': len(kernel_size_1d)*[256*width],
                    'kernel_size': kernel_size_1d,
                    'residual_connections': residual_connections_1d,
                    'norm': 'batch',
                    'norm_kwargs': {'eps': 1e-3},
                    'activation_fn': 'relu',
                    'pre_activation': True,
                    'dropout': .0,
                    'output_layer': False,
                },
            },
            'rnn_fwd': {
                'rnn': {
                    'hidden_size': 256*width,
                    'num_layers': 2,
                    'dropout': .0,
                },
                'output_net': {
                    'out_channels': [
                        256*width,
                        num_events
                    ],
                    'kernel_size': 1,
                    'norm': 'batch',
                    'norm_kwargs': {'eps': 1e-3},
                    'activation_fn': 'relu',
                    'dropout': .0,
                }
            },
            'labelwise_metrics': ('fscore_weak',),
            'strong_fwd_bwd_loss_weight': strong_fwd_bwd_loss_weight,
        },
        'optimizer': {
            'factory': Adam,
            'lr': lr,
            'gradient_clipping': gradient_clipping,
            # 'weight_decay': 1e-6,
        },
        'summary_trigger': (summary_interval, 'iteration'),
        'checkpoint_trigger': (checkpoint_interval, 'iteration'),
        'stop_trigger': (num_iterations, 'iteration'),
        'storage_dir': storage_dir,
    }
    use_transformer = False
    if use_transformer:
        trainer['model']['rnn_fwd']['factory'] = TransformerEncoder
        trainer['model']['rnn_fwd']['rnn']['hidden_size'] = 256*width
        trainer['model']['rnn_fwd']['rnn']['d_ff'] = 1024*width
        trainer['model']['rnn_fwd']['rnn']['num_layers'] = 6  # * (1 + (net_config == 'deep'))
        trainer['model']['rnn_fwd']['rnn']['dropout'] = 0.2

    Trainer.get_config(trainer)
    device = None
    track_emissions = False
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def train(
        _run, debug, resume, delay,
        data_provider, filter_desed_test_clips, trainer, lr_rampup_steps,
        n_back_off, back_off_patience, lr_decay_steps, lr_decay_factor,
        early_stopping_patience,
        init_ckpt_path, frozen_cnn_2d_layers,
        frozen_cnn_1d_layers, freeze_norm_stats,
        validation_set_name, validation_ground_truth_filepath,
        eval_set_name, eval_ground_truth_filepath,
        device, track_emissions, hyper_params_tuning_batch_size,
):
    print()
    print('##### Training #####')
    print()
    print_config(_run)
    assert (n_back_off == 0) or (len(lr_decay_steps) == 0), (n_back_off, lr_decay_steps)
    if delay > 0:
        print(f'Sleep for {delay} seconds.')
        time.sleep(delay)

    data_provider = DataProvider.from_config(data_provider)
    data_provider.train_transform.label_encoder.initialize_labels(
        dataset=data_provider.db.get_dataset(list(filter(
            lambda key: data_provider.train_set[key] > 0,
            data_provider.train_set.keys()
        ))),
        verbose=True
    )
    data_provider.test_transform.label_encoder.initialize_labels()
    trainer = Trainer.from_config(trainer)
    trainer.model.label_mapping = []
    for idx, label in sorted(data_provider.train_transform.label_encoder.inverse_label_mapping.items()):
        assert idx == len(trainer.model.label_mapping), (idx, label, len(trainer.model.label_mapping))
        trainer.model.label_mapping.append(label.replace(', ', '__').replace(' ', '').replace('(', '_').replace(')', '_').replace("'", ''))
    print('Params', sum(p.numel() for p in trainer.model.parameters()))
    print('CNN Params', sum(p.numel() for p in trainer.model.cnn.parameters()))

    if init_ckpt_path is not None:
        print('Load init params')
        state_dict = deflatten(torch.load(init_ckpt_path, map_location='cpu')['model'], maxdepth=2)
        trainer.model.cnn.load_state_dict(flatten(state_dict['cnn']))
        trainer.model.rnn_fwd.rnn.load_state_dict(state_dict['rnn_fwd']['rnn'])
        trainer.model.rnn_bwd.rnn.load_state_dict(state_dict['rnn_bwd']['rnn'])
        # pop output layer from checkpoint
        param_keys = sorted(state_dict['rnn_fwd']['output_net'].keys())
        layer_idx = [key.split('.')[1] for key in param_keys]
        last_layer_idx = layer_idx[-1]
        for key, layer_idx in zip(param_keys, layer_idx):
            if layer_idx == last_layer_idx:
                state_dict['rnn_fwd']['output_net'].pop(key)
                state_dict['rnn_bwd']['output_net'].pop(key)
        trainer.model.rnn_fwd.output_net.load_state_dict(state_dict['rnn_fwd']['output_net'], strict=False)
        trainer.model.rnn_bwd.output_net.load_state_dict(state_dict['rnn_bwd']['output_net'], strict=False)
    if frozen_cnn_2d_layers:
        print(f'Freeze {frozen_cnn_2d_layers} cnn_2d layers')
        trainer.model.cnn.cnn_2d.freeze(
            frozen_cnn_2d_layers, freeze_norm_stats=freeze_norm_stats)
    if frozen_cnn_1d_layers:
        print(f'Freeze {frozen_cnn_1d_layers} cnn_1d layers')
        trainer.model.cnn.cnn_1d.freeze(
            frozen_cnn_1d_layers, freeze_norm_stats=freeze_norm_stats)

    if filter_desed_test_clips:
        with (database_jsons_dir / 'desed.json').open() as fid:
            desed_json = json.load(fid)
        filter_example_ids = {
            clip_id.rsplit('_', maxsplit=2)[0][1:]
            for clip_id in (
                list(desed_json['datasets']['validation'].keys())
                + list(desed_json['datasets']['eval_public'].keys())
            )
        }
    else:
        filter_example_ids = None
    train_set = data_provider.get_train_set(filter_example_ids=filter_example_ids)
    validate_set = data_provider.get_validate_set()

    if validate_set is not None:
        trainer.test_run(train_set, validate_set)
        trainer.register_validation_hook(
            validate_set, metric='macro_fscore_weak', maximize=True,
            back_off_patience=back_off_patience,
            n_back_off=n_back_off,
            lr_update_factor=lr_decay_factor,
            early_stopping_patience=early_stopping_patience,
        )

    breakpoints = []
    if lr_rampup_steps is not None:
        breakpoints += [(0, 0.), (lr_rampup_steps, 1.)]
    for i, lr_decay_step in enumerate(lr_decay_steps):
        breakpoints += [(lr_decay_step, lr_decay_factor**i), (lr_decay_step, lr_decay_factor**(i+1))]
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
    trainer.train(
        train_set, resume=resume, device=device,
        track_emissions=track_emissions,
    )

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
                        'batch_size': hyper_params_tuning_batch_size,
                    }
                },
            }
        )
