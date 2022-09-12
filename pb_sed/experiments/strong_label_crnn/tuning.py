import os
import numpy as np
from pathlib import Path
from functools import partial
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver
from sacred.commands import print_config

from paderbox.utils.timer import timeStamped
from paderbox.io.json_module import load_json, dump_json
from sed_scores_eval import collar_based

from pb_sed.paths import storage_root
from pb_sed.models import weak_label
from pb_sed.models import strong_label
from pb_sed.models import base
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.experiments.weak_label_crnn.inference import tagging
from pb_sed.experiments.strong_label_crnn.inference import ex as evaluation


ex_name = 'strong_label_crnn_hyper_params'
ex = Exp(ex_name)


@ex.config
def config():
    debug = False
    timestamp = timeStamped('')[1:] + ('_debug' if debug else '')

    weak_label_crnn_hyper_params_dir = ''
    assert len(weak_label_crnn_hyper_params_dir) > 0, 'Set weak_label_crnn_hyper_params_dir on the command line.'
    weak_label_crnn_tuning_config = load_json(Path(weak_label_crnn_hyper_params_dir) / '1' / 'config.json')
    weak_label_crnn_dirs = weak_label_crnn_tuning_config['crnn_dirs']
    assert len(weak_label_crnn_dirs) > 0, 'weak_label_crnn_dirs must not be empty.'
    weak_label_crnn_checkpoints = weak_label_crnn_tuning_config['crnn_checkpoints']
    del weak_label_crnn_tuning_config

    strong_label_crnn_group_dir = ''
    if isinstance(strong_label_crnn_group_dir, list):
        strong_label_crnn_dirs = sorted([
            str(d) for g in strong_label_crnn_group_dir for d in Path(g).glob('202*') if d.is_dir()
        ])
    else:
        strong_label_crnn_dirs = sorted([str(d) for d in Path(strong_label_crnn_group_dir).glob('202*') if d.is_dir()])
    assert len(strong_label_crnn_dirs) > 0, 'strong_label_crnn_dirs must not be empty.'
    strong_label_crnn_checkpoints = 'ckpt_best_macro_fscore_strong.pth'
    strong_crnn_config = load_json(Path(strong_label_crnn_dirs[0]) / '1' / 'config.json')
    data_provider = strong_crnn_config['data_provider']
    database_name = strong_crnn_config.get('database_name', 'desed')
    storage_dir = str(storage_root / 'strong_label_crnn' / database_name / 'hyper_params' / timestamp)
    assert not Path(storage_dir).exists()
    del strong_crnn_config
    data_provider['min_audio_length'] = .01
    data_provider['cached_datasets'] = None

    device = 0

    validation_set_name = 'validation'
    validation_ground_truth_filepath = None
    eval_set_name = 'eval_public'
    eval_ground_truth_filepath = None

    medfilt_lengths = [31] if debug else [301, 251, 201, 151, 101, 81, 61, 51, 41, 31, 21, 11]

    ex.observers.append(FileStorageObserver.create(storage_dir))


@ex.automain
def main(
        _run, storage_dir, debug,
        weak_label_crnn_hyper_params_dir, weak_label_crnn_dirs, weak_label_crnn_checkpoints,
        strong_label_crnn_dirs, strong_label_crnn_checkpoints,
        data_provider, validation_set_name, validation_ground_truth_filepath,
        eval_set_name, eval_ground_truth_filepath,
        medfilt_lengths, device
):
    print()
    print('##### Tuning #####')
    print()
    print_config(_run)
    print(storage_dir)
    storage_dir = Path(storage_dir)

    if not isinstance(weak_label_crnn_checkpoints, list):
        assert isinstance(weak_label_crnn_checkpoints, str), weak_label_crnn_checkpoints
        weak_label_crnn_checkpoints = len(weak_label_crnn_dirs) * [weak_label_crnn_checkpoints]
    weak_label_crnns = [
        weak_label.CRNN.from_storage_dir(
            storage_dir=crnn_dir, config_name='1/config.json',
            checkpoint_name=crnn_checkpoint
        )
        for crnn_dir, crnn_checkpoint in zip(weak_label_crnn_dirs, weak_label_crnn_checkpoints)
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
        if 'audio_length' in example
    }

    timestamps = {
        audio_id: np.array([0., audio_durations[audio_id]])
        for audio_id in audio_durations
    }
    tags, tagging_scores, _ = tagging(
        weak_label_crnns, dataset, device, timestamps, event_classes,
        weak_label_crnn_hyper_params_dir, None, None,
    )

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
    metrics = {
        'f': partial(
            base.f_collar, ground_truth=validation_ground_truth_filepath,
            return_onset_offset_bias=True, num_jobs=8,
            **collar_based_params,
        ),
        'auc1': partial(
            base.psd_auc, ground_truth=validation_ground_truth_filepath,
            audio_durations=audio_durations, num_jobs=8,
            **psds_scenario_1,
        ),
        'auc2': partial(
            base.psd_auc, ground_truth=validation_ground_truth_filepath,
            audio_durations=audio_durations, num_jobs=8,
            **psds_scenario_2,
        )
    }

    if not isinstance(strong_label_crnn_checkpoints, list):
        assert isinstance(strong_label_crnn_checkpoints, str), strong_label_crnn_checkpoints
        strong_label_crnn_checkpoints = len(strong_label_crnn_dirs) * [strong_label_crnn_checkpoints]
    strong_label_crnns = [
        strong_label.CRNN.from_storage_dir(
            storage_dir=crnn_dir, config_name='1/config.json',
            checkpoint_name=crnn_checkpoint
        )
        for crnn_dir, crnn_checkpoint in zip(
            strong_label_crnn_dirs, strong_label_crnn_checkpoints)
    ]

    def add_tag_condition(example):
        example["tag_condition"] = np.array([
            tags[example_id] for example_id in example["example_id"]])
        return example

    timestamps = np.arange(0, 10000) * frame_shift
    leaderboard = strong_label.crnn.tune_sound_event_detection(
        strong_label_crnns, dataset.map(add_tag_condition), device, timestamps,
        event_classes, tags, metrics,
        tag_masking={'f': True, 'auc1': '?', 'auc2': '?'},
        medfilt_lengths=medfilt_lengths,
    )
    dump_json(leaderboard['f'][1], storage_dir / f'sed_hyper_params_f.json')
    f, p, r, thresholds, _ = collar_based.best_fscore(
        scores=leaderboard['auc1'][2],
        ground_truth=validation_ground_truth_filepath,
        **collar_based_params, num_jobs=8
    )
    for event_class in thresholds:
        leaderboard['auc1'][1][event_class]['threshold'] = thresholds[event_class]
    dump_json(leaderboard['auc1'][1], storage_dir / 'sed_hyper_params_psds1.json')
    f, p, r, thresholds, _ = collar_based.best_fscore(
        scores=leaderboard['auc2'][2],
        ground_truth=validation_ground_truth_filepath,
        **collar_based_params, num_jobs=8
    )
    for event_class in thresholds:
        leaderboard['auc2'][1][event_class]['threshold'] = thresholds[event_class]
    dump_json(leaderboard['auc2'][1], storage_dir / 'sed_hyper_params_psds2.json')
    for crnn_dir in strong_label_crnn_dirs:
        tuning_dir = Path(crnn_dir) / 'hyper_params'
        os.makedirs(str(tuning_dir), exist_ok=True)
        (tuning_dir / storage_dir.name).symlink_to(storage_dir)
    print(storage_dir)

    if eval_set_name:
        evaluation.run(
            config_updates={
                'debug': debug,
                'strong_label_crnn_hyper_params_dir': str(storage_dir),
                'dataset_name': eval_set_name,
                'ground_truth_filepath': eval_ground_truth_filepath,
            },
        )
