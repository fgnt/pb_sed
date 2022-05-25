import os
import numpy as np
from pathlib import Path
from functools import partial
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver
from sacred.commands import print_config

from paderbox.utils.timer import timeStamped
from paderbox.io.json_module import load_json, dump_json
from sed_scores_eval import io

from pb_sed.paths import storage_root
from pb_sed.models import weak_label
from pb_sed.models import base
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.experiments.weak_label_crnn.inference import ex as evaluation


ex_name = 'weak_label_crnn_hyper_params'
ex = Exp(ex_name)


@ex.config
def config():
    debug = False
    timestamp = timeStamped('')[1:] + ('_debug' if debug else '')
    storage_dir = str(storage_root / ex_name / timestamp)
    assert not Path(storage_dir).exists()

    group_dir = ''
    if isinstance(group_dir, list):
        crnn_dirs = sorted([
            str(d) for g in group_dir for d in Path(g).glob('202*') if d.is_dir()
        ])
    else:
        crnn_dirs = sorted([str(d) for d in Path(group_dir).glob('202*') if d.is_dir()])
    assert len(crnn_dirs) > 0, 'crnn_dirs must not be empty.'
    crnn_checkpoints = 'ckpt_best_macro_fscore_weak.pth'
    crnn_config = load_json(Path(crnn_dirs[0]) / '1' / 'config.json')
    data_provider = crnn_config['data_provider']
    del crnn_config
    data_provider['min_audio_length'] = .01

    device = 0

    validation_set_name = 'validation'
    validation_ground_truth_filepath = None
    eval_set_name = 'eval_public'
    eval_ground_truth_filepath = None

    boundaries_filter_lengths = [20] if debug else [100, 80, 60, 50, 40, 30, 20, 10, 0]

    detection_window_lengths_scenario_1 = [21] if debug else [51, 41, 31, 21, 11]
    detection_window_shift_scenario_1 = 1
    detection_medfilt_lengths_scenario_1 = [21] if debug else [101, 81, 61, 51, 41, 31, 21, 11]

    detection_window_lengths_scenario_2 = [250]
    detection_window_shift_scenario_2 = 250
    detection_medfilt_lengths_scenario_2 = [1]

    ex.observers.append(FileStorageObserver.create(storage_dir))


@ex.automain
def main(
        _run,
        storage_dir, debug,
        crnn_dirs, crnn_checkpoints, data_provider,
        validation_set_name, validation_ground_truth_filepath,
        eval_set_name, eval_ground_truth_filepath,
        boundaries_filter_lengths,
        detection_window_lengths_scenario_1,
        detection_window_shift_scenario_1,
        detection_medfilt_lengths_scenario_1,
        detection_window_lengths_scenario_2,
        detection_window_shift_scenario_2,
        detection_medfilt_lengths_scenario_2,
        device
):
    print()
    print('##### Tuning #####')
    print()
    print_config(_run)
    print(storage_dir)
    storage_dir = Path(storage_dir)

    boundaries_collar_based_params = {
        'onset_collar': .5,
        'offset_collar': .5,
        'offset_collar_rate': .0,
        'min_precision': .8,
    }
    collar_based_params = {
        'onset_collar': .2,
        'offset_collar': .2,
        'offset_collar_rate': .2,
    }
    psds_scenario_1 = {
        'dtc_threshold': 0.7,
        'gtc_threshold': 0.7,
        'cttc_threshold': None,
        'alpha_ct': .0,
        'alpha_st': 1.,
    }
    psds_scenario_2 = {
        'dtc_threshold': 0.1,
        'gtc_threshold': 0.1,
        'cttc_threshold': 0.3,
        'alpha_ct': .5,
        'alpha_st': 1.,
    }

    if not isinstance(crnn_checkpoints, list):
        assert isinstance(crnn_checkpoints, str), crnn_checkpoints
        crnn_checkpoints = len(crnn_dirs) * [crnn_checkpoints]
    crnns = [
        weak_label.CRNN.from_storage_dir(
            storage_dir=crnn_dir, config_name='1/config.json',
            checkpoint_name=crnn_checkpoint
        )
        for crnn_dir, crnn_checkpoint in zip(crnn_dirs, crnn_checkpoints)
    ]
    data_provider = DESEDProvider.from_config(data_provider)
    data_provider.test_transform.label_encoder.initialize_labels()
    event_classes = data_provider.test_transform.label_encoder.inverse_label_mapping
    event_classes = [event_classes[i] for i in range(len(event_classes))]
    frame_shift = data_provider.test_transform.stft.shift
    frame_shift /= data_provider.audio_reader.target_sample_rate

    if validation_set_name == 'validation' and not validation_ground_truth_filepath:
        database_root = Path(data_provider.get_raw('validation')[0]['audio_path']).parent.parent.parent.parent
        validation_ground_truth_filepath = database_root / 'metadata' / 'validation' / 'validation.tsv'
    elif validation_set_name == 'eval_public' and not validation_ground_truth_filepath:
        database_root = Path(data_provider.get_raw('eval_public')[0]['audio_path']).parent.parent.parent.parent
        validation_ground_truth_filepath = database_root / 'metadata' / 'eval' / 'public.tsv'
    assert isinstance(validation_ground_truth_filepath, (str, Path)) and Path(validation_ground_truth_filepath).exists(), validation_ground_truth_filepath

    dataset = data_provider.get_dataset(validation_set_name)
    audio_durations = {
        example['example_id']: example['audio_length']
        for example in data_provider.db.get_dataset(validation_set_name)
    }

    timestamps = {
        audio_id: np.array([0., audio_durations[audio_id]])
        for audio_id in audio_durations
    }
    metrics = {
        'f': partial(base.f_tag, ground_truth=validation_ground_truth_filepath, num_jobs=8)
    }
    leaderboard = weak_label.crnn.tune_tagging(
        crnns, dataset, device, timestamps, event_classes, metrics,
        storage_dir=storage_dir
    )
    io.write_score_transform(
        scores=leaderboard['f'][2],
        ground_truth=validation_ground_truth_filepath,
        filepath=storage_dir / f'tagging_score_transform.tsv',
    )
    _, hyper_params, tagging_scores = leaderboard['f']
    tagging_thresholds = np.array([
        hyper_params[event_class]['threshold'] for event_class in event_classes
    ])
    tags = {
        audio_id: tagging_scores[audio_id][event_classes].to_numpy() > tagging_thresholds
        for audio_id in tagging_scores
    }

    boundaries_ground_truth = base.boundaries_from_events(validation_ground_truth_filepath)
    timestamps = np.arange(0, 10000) * frame_shift
    metrics = {
        'f': partial(
            base.f_collar, ground_truth=boundaries_ground_truth,
            return_onset_offset_bias=True, num_jobs=8,
            **boundaries_collar_based_params,
        ),
    }
    weak_label.crnn.tune_boundary_detection(
        crnns, dataset, device, timestamps, event_classes, tags, metrics,
        tag_masking=True, stepfilt_lengths=boundaries_filter_lengths,
        storage_dir=storage_dir
    )

    metrics = {
        'f': partial(
            base.f_collar, ground_truth=validation_ground_truth_filepath,
            return_onset_offset_bias=True, num_jobs=8, **collar_based_params,
        ),
        'auc': partial(
            base.psd_auc, ground_truth=validation_ground_truth_filepath,
            audio_durations=audio_durations, num_jobs=8,
            **psds_scenario_1,
        ),
    }
    leaderboard = weak_label.crnn.tune_sound_event_detection(
        crnns, dataset, device, timestamps, event_classes, tags, metrics,
        tag_masking={'f': True, 'auc': '?'},
        window_lengths=detection_window_lengths_scenario_1,
        window_shift=detection_window_shift_scenario_1,
        medfilt_lengths=detection_medfilt_lengths_scenario_1,
    )
    dump_json(leaderboard['f'][1], storage_dir / f'sed_hyper_params_f.json')
    io.write_score_transform(
        scores=leaderboard['f'][2],
        ground_truth=validation_ground_truth_filepath,
        filepath=storage_dir / f'sed_score_transform_f.tsv',
    )
    dump_json(leaderboard['auc'][1], storage_dir / 'sed_hyper_params_psds1.json')
    io.write_score_transform(
        scores=leaderboard['auc'][2],
        ground_truth=validation_ground_truth_filepath,
        filepath=storage_dir / f'sed_score_transform_psds1.tsv',
    )

    metrics = {
        'auc': partial(
            base.psd_auc, ground_truth=validation_ground_truth_filepath,
            audio_durations=audio_durations, num_jobs=8,
            **psds_scenario_2,
        )
    }
    leaderboard = weak_label.crnn.tune_sound_event_detection(
        crnns, dataset, device, timestamps, event_classes, tags, metrics,
        tag_masking=False,
        window_lengths=detection_window_lengths_scenario_2,
        window_shift=detection_window_shift_scenario_2,
        medfilt_lengths=detection_medfilt_lengths_scenario_2,
    )
    dump_json(
        leaderboard['auc'][1], storage_dir / 'sed_hyper_params_psds2.json')
    io.write_score_transform(
        scores=leaderboard['auc'][2],
        ground_truth=validation_ground_truth_filepath,
        filepath=storage_dir / f'sed_score_transform_psds2.tsv',
    )
    for crnn_dir in crnn_dirs:
        tuning_dir = Path(crnn_dir) / 'hyper_params'
        os.makedirs(str(tuning_dir), exist_ok=True)
        (tuning_dir / storage_dir.name).symlink_to(storage_dir)
    print(storage_dir)

    if eval_set_name:
        evaluation.run(
            config_updates={
                'debug': debug,
                'hyper_params_dir': str(storage_dir),
                'dataset_name': eval_set_name,
                'ground_truth_filepath': eval_ground_truth_filepath,
            },
        )
