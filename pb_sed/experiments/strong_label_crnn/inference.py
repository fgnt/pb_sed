import os
import numpy as np
from pathlib import Path
from copy import deepcopy
from functools import partial
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from codecarbon import EmissionsTracker

from paderbox.utils.timer import timeStamped
from paderbox.io.json_module import load_json, dump_json
from sed_scores_eval import intersection_based, collar_based
from sed_scores_eval.base_modules.detection import scores_to_event_list
from sed_scores_eval.utils.auc import staircase_auc
from sed_scores_eval import io

from pb_sed.paths import storage_root
from pb_sed.models import weak_label
from pb_sed.models import strong_label
from pb_sed.models import base
from pb_sed.database.desed.provider import DESEDProvider
from pb_sed.experiments.weak_label_crnn.inference import tagging
from pb_sed.utils.segment import segment_batch, merge_segments

ex_name = 'strong_label_crnn_inference'
ex = Exp(ex_name)


@ex.config
def config():
    debug = False
    timestamp = timeStamped('')[1:] + ('_debug' if debug else '')

    strong_label_crnn_hyper_params_dir = ''
    assert len(strong_label_crnn_hyper_params_dir) > 0, 'Set strong_label_crnn_hyper_params_dir on the command line.'
    strong_label_crnn_tuning_config = load_json(Path(strong_label_crnn_hyper_params_dir) / '1' / 'config.json')
    strong_label_crnn_dirs = strong_label_crnn_tuning_config['strong_label_crnn_dirs']
    assert len(strong_label_crnn_dirs) > 0, 'strong_label_crnn_dirs must not be empty.'
    strong_label_crnn_checkpoints = strong_label_crnn_tuning_config['strong_label_crnn_checkpoints']
    data_provider = strong_label_crnn_tuning_config['data_provider']
    database_name = strong_label_crnn_tuning_config['database_name']
    storage_dir = str(storage_root / 'strong_label_crnn' / database_name / 'inference' / timestamp)
    assert not Path(storage_dir).exists()

    weak_label_crnn_hyper_params_dir = strong_label_crnn_tuning_config['weak_label_crnn_hyper_params_dir']
    assert len(weak_label_crnn_hyper_params_dir) > 0, 'Set weak_label_crnn_hyper_params_dir on the command line.'
    weak_label_crnn_tuning_config = load_json(Path(weak_label_crnn_hyper_params_dir) / '1' / 'config.json')
    weak_label_crnn_dirs = weak_label_crnn_tuning_config['crnn_dirs']
    assert len(weak_label_crnn_dirs) > 0, 'weak_label_crnn_dirs must not be empty.'
    weak_label_crnn_checkpoints = weak_label_crnn_tuning_config['crnn_checkpoints']

    del strong_label_crnn_tuning_config
    del weak_label_crnn_tuning_config

    sed_hyper_params_name = ['f', 'psds1', 'psds2']

    device = 0

    dataset_name = 'eval_public'
    ground_truth_filepath = None

    max_segment_length = None
    if max_segment_length is None:
        segment_overlap = None
    else:
        segment_overlap = 100
    save_scores = False
    save_detections = False

    weak_pseudo_labeling = False
    strong_pseudo_labeling = False
    pseudo_labelled_dataset_name = dataset_name

    pseudo_widening = .0

    ex.observers.append(FileStorageObserver.create(storage_dir))


def sound_event_detection(
        crnns, dataset, device, timestamps, event_classes, tags,
        hyper_params_dir, hyper_params_name,
        ground_truth, audio_durations,
        collar_based_params=(), psds_params=(),
        max_segment_length=None, segment_overlap=None, pseudo_widening=.0,
        score_storage_dir=None, detection_storage_dir=None,
):
    print()
    print('Sound Event Detection')
    if isinstance(hyper_params_name, (str, Path)):
        hyper_params_name = [hyper_params_name]
    assert isinstance(hyper_params_name, (list, tuple))
    hyper_params = [
        load_json(Path(hyper_params_dir) / f'sed_hyper_params_{name}.json')
        for name in hyper_params_name
    ]

    if isinstance(score_storage_dir, (list, tuple)):
        assert len(score_storage_dir) == len(hyper_params), (len(score_storage_dir), len(hyper_params))
    elif isinstance(score_storage_dir, (str, Path)):
        score_storage_dir = [
            Path(score_storage_dir) / name for name in hyper_params_name
        ]
    elif score_storage_dir is not None:
        raise ValueError('score_storage_dir must be list, str, Path or None.')

    if isinstance(detection_storage_dir, (list, tuple)):
        assert len(detection_storage_dir) == len(hyper_params), (len(detection_storage_dir), len(hyper_params))
    elif isinstance(detection_storage_dir, (str, Path)):
        detection_storage_dir = [
            Path(detection_storage_dir) / name for name in hyper_params_name
        ]
    elif detection_storage_dir is not None:
        raise ValueError('detection_storage_dir must be list, str, Path or None.')

    medfilt_lengths = np.zeros((len(hyper_params), len(event_classes)))
    tag_masked = np.zeros((len(hyper_params), len(event_classes)))
    for i, hyper_params_i in enumerate(hyper_params):
        for j, event_class in enumerate(event_classes):
            medfilt_lengths[i, j] = hyper_params_i[event_class]['medfilt_length']
            tag_masked[i, j] = hyper_params_i[event_class]['tag_masked']
    detection_scores = base.sound_event_detection(
        crnns, dataset, device,
        medfilt_length=medfilt_lengths, apply_mask=tag_masked, masks=tags,
        timestamps=timestamps, event_classes=event_classes,
        max_segment_length=max_segment_length, segment_overlap=segment_overlap,
        merge_score_segments=True, score_segment_overlap=segment_overlap,
        score_storage_dir=score_storage_dir,
    )
    event_detections = []
    results = []
    for i, name in enumerate(hyper_params_name):
        if ground_truth:
            print()
            print(name)
        results.append({})
        if detection_storage_dir and detection_storage_dir[i]:
            io.write_detections_for_multiple_thresholds(
                detection_scores[i], thresholds=np.linspace(.01, .99, 50),
                dir_path=detection_storage_dir[i],
            )
        if 'threshold' in hyper_params[i][event_classes[0]]:
            thresholds = {
                event_class: hyper_params[i][event_class]['threshold']
                for event_class in event_classes
            }
            event_detections.append(
                scores_to_event_list(
                    detection_scores[i], thresholds, event_classes=event_classes)
            )
            if detection_storage_dir and detection_storage_dir[i]:
                io.write_detection(
                    detection_scores[i], thresholds,
                    Path(detection_storage_dir[i]) / 'cbf.tsv',
                )
            if ground_truth and collar_based_params:
                f, p, r, stats = collar_based.fscore(
                    detection_scores[i], ground_truth, thresholds,
                    **collar_based_params,
                    return_onset_offset_dist_sum=True, num_jobs=8,
                )
                print('f', f)
                print('p', p)
                print('r', r)
                for key in f:
                    results[-1].update({
                        f'{key}_f': f[key],
                        f'{key}_p': p[key],
                        f'{key}_r': r[key],
                    })
                    if key in stats:
                        results[-1].update({
                            f'{key}_onset_bias': stats[key]['onset_dist_sum'] / max(stats[key]['tps'], 1),
                            f'{key}_offset_bias': stats[key]['offset_dist_sum'] / max(stats[key]['tps'], 1),
                        })

            for clip_id in event_detections[-1]:
                events_in_clip = []
                for onset, offset, event_label in event_detections[-1][clip_id]:
                    onset = max(onset - pseudo_widening - hyper_params[i][event_label].get('onset_bias', 0), 0)
                    offset = offset + pseudo_widening - hyper_params[i][event_label].get('offset_bias', 0)
                    if offset > onset:
                        events_in_clip.append((onset, offset, event_label))
                event_detections[-1][clip_id] = events_in_clip
        else:
            event_detections.append(None)
        if ground_truth:
            if not isinstance(psds_params, (tuple, list)):
                psds_params = [psds_params]
            for j in range(len(psds_params)):
                psds, psd_roc, classwise_rocs = intersection_based.psds(
                    detection_scores[i], ground_truth, audio_durations,
                    **psds_params[j], num_jobs=8,
                )
                print(f'psds[{j}]', psds)
                results[-1][f'psds[{j}]'] = psds
                for event_class, (tpr, efpr, *_) in classwise_rocs.items():
                    results[-1][f'{event_class}_auc[{j}]'] = staircase_auc(
                        tpr, efpr, psds_params[j].get('max_efpr', 100))
                if score_storage_dir and score_storage_dir[i] is not None:
                    psds, psd_roc, classwise_rocs = intersection_based.psds(
                        score_storage_dir[i], ground_truth, audio_durations,
                        **psds_params[j], num_jobs=8,
                    )
                    print(f'psds[{j}] (from files)', psds)
                psds, psd_roc, classwise_rocs = intersection_based.reference.approximate_psds(
                    detection_scores[i], ground_truth, audio_durations,
                    **psds_params[j], thresholds=np.linspace(.01, .99, 50),
                )
                print(f'approx_psds[{j}]', psds)
                results[-1][f'approx_psds[{j}]'] = psds
                for event_class, (tpr, efpr, *_) in classwise_rocs.items():
                    results[-1][f'{event_class}_approx_auc[{j}]'] = staircase_auc(
                        tpr, efpr, psds_params[j].get('max_efpr', 100))
                if detection_storage_dir and detection_storage_dir[i] is not None:
                    psds, psd_roc, classwise_rocs = intersection_based.reference.approximate_psds_from_detections_dir(
                        detection_storage_dir[i], ground_truth, audio_durations,
                        **psds_params[j], thresholds=np.linspace(.01, .99, 50),
                    )
                    print(f'approx_psds[{j}] (from files)', psds)
    return event_detections, results


@ex.automain
def main(
        _run,
        storage_dir, strong_label_crnn_hyper_params_dir, sed_hyper_params_name,
        strong_label_crnn_dirs, strong_label_crnn_checkpoints,
        weak_label_crnn_hyper_params_dir, weak_label_crnn_dirs, weak_label_crnn_checkpoints,
        device, data_provider, dataset_name, ground_truth_filepath,
        save_scores, save_detections, max_segment_length, segment_overlap,
        strong_pseudo_labeling, pseudo_widening, pseudo_labelled_dataset_name,
):
    print()
    print('##### Inference #####')
    print()
    print_config(_run)
    print(storage_dir)
    emissions_tracker = EmissionsTracker(
        output_dir=storage_dir, on_csv_write="update", log_level='error')
    emissions_tracker.start()
    storage_dir = Path(storage_dir)

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
    print('Weak Label CRNN Params', sum([p.numel() for crnn in weak_label_crnns for p in crnn.parameters()]))
    print('Weak Label CNN2d Params', sum([p.numel() for crnn in weak_label_crnns for p in crnn.cnn.cnn_2d.parameters()]))
    if not isinstance(strong_label_crnn_checkpoints, list):
        assert isinstance(strong_label_crnn_checkpoints, str), strong_label_crnn_checkpoints
        strong_label_crnn_checkpoints = len(strong_label_crnn_dirs) * [strong_label_crnn_checkpoints]
    strong_label_crnns = [
        strong_label.CRNN.from_storage_dir(
            storage_dir=crnn_dir, config_name='1/config.json',
            checkpoint_name=crnn_checkpoint
        )
        for crnn_dir, crnn_checkpoint in zip(strong_label_crnn_dirs, strong_label_crnn_checkpoints)
    ]
    print('Strong Label CRNN Params', sum([p.numel() for crnn in strong_label_crnns for p in crnn.parameters()]))
    print('Strong Label CNN2d Params', sum([p.numel() for crnn in strong_label_crnns for p in crnn.cnn.cnn_2d.parameters()]))
    data_provider = DESEDProvider.from_config(data_provider)
    data_provider.test_transform.label_encoder.initialize_labels()
    event_classes = data_provider.test_transform.label_encoder.inverse_label_mapping
    event_classes = [event_classes[i] for i in range(len(event_classes))]
    frame_shift = data_provider.test_transform.stft.shift
    frame_shift /= data_provider.audio_reader.target_sample_rate

    if not isinstance(dataset_name, list):
        dataset_name = [dataset_name]
    if ground_truth_filepath is None:
        ground_truth_filepath = len(dataset_name)*[ground_truth_filepath]
    elif isinstance(ground_truth_filepath, (str, Path)):
        ground_truth_filepath = [ground_truth_filepath]
    assert len(ground_truth_filepath) == len(dataset_name)
    if not isinstance(strong_pseudo_labeling, list):
        strong_pseudo_labeling = len(dataset_name)*[strong_pseudo_labeling]
    assert len(strong_pseudo_labeling) == len(dataset_name)
    if not isinstance(pseudo_labelled_dataset_name, list):
        pseudo_labelled_dataset_name = [pseudo_labelled_dataset_name]
    assert len(pseudo_labelled_dataset_name) == len(dataset_name)

    database = deepcopy(data_provider.db.data)
    for i in range(len(dataset_name)):
        print()
        print(dataset_name[i])
        if dataset_name[i] == 'eval_public' and not ground_truth_filepath[i]:
            database_root = Path(data_provider.get_raw('eval_public')[0]['audio_path']).parent.parent.parent.parent
            ground_truth_filepath[i] = database_root / 'metadata' / 'eval' / 'public.tsv'
        elif dataset_name[i] == 'validation' and not ground_truth_filepath[i]:
            database_root = Path(data_provider.get_raw('validation')[0]['audio_path']).parent.parent.parent.parent
            ground_truth_filepath[i] = database_root / 'metadata' / 'validation' / 'validation.tsv'

        dataset = data_provider.get_dataset(dataset_name[i])
        audio_durations = {
            example['example_id']: example['audio_length']
            for example in data_provider.db.get_dataset(dataset_name[i])
            if 'audio_length' in example
        }

        score_storage_dir = storage_dir / 'scores' / dataset_name[i]
        detection_storage_dir = storage_dir / 'detections' / dataset_name[i]

        if max_segment_length is None:
            timestamps = {
                audio_id: np.array([0., audio_durations[audio_id]])
                for audio_id in audio_durations
            }
        else:
            timestamps = {}
            for audio_id in audio_durations:
                ts = np.arange(
                    (2+max_segment_length)*frame_shift,
                    audio_durations[audio_id],
                    (max_segment_length-segment_overlap)*frame_shift
                )
                timestamps[audio_id] = np.concatenate((
                    [0.], ts-segment_overlap/2*frame_shift,
                    [audio_durations[audio_id]]
                ))
        if max_segment_length is not None:
            dataset = dataset.map(partial(
                segment_batch,
                max_length=max_segment_length,
                overlap=segment_overlap
            )).unbatch()
        tags, tagging_scores, _ = tagging(
            weak_label_crnns, dataset, device, timestamps, event_classes,
            weak_label_crnn_hyper_params_dir, None, None,
        )

        def add_tag_condition(example):
            example["tag_condition"] = np.array([
                tags[example_id] for example_id in example["example_id"]])
            return example

        dataset = dataset.map(add_tag_condition)

        timestamps = np.round(np.arange(0, 100000) * frame_shift, decimals=6)
        if not isinstance(sed_hyper_params_name, (list, tuple)):
            sed_hyper_params_name = [sed_hyper_params_name]
        events, sed_results = sound_event_detection(
            strong_label_crnns, dataset, device, timestamps, event_classes, tags,
            strong_label_crnn_hyper_params_dir, sed_hyper_params_name,
            ground_truth_filepath[i], audio_durations,
            collar_based_params, [psds_scenario_1, psds_scenario_2],
            max_segment_length=max_segment_length,
            segment_overlap=segment_overlap,
            pseudo_widening=pseudo_widening,
            score_storage_dir=[score_storage_dir / name for name in sed_hyper_params_name]
            if save_scores else None,
            detection_storage_dir=[detection_storage_dir / name for name in sed_hyper_params_name]
            if save_detections else None,
        )
        for j, sed_results_j in enumerate(sed_results):
            if sed_results_j:
                dump_json(
                    sed_results_j,
                    storage_dir / f'sed_{sed_hyper_params_name[j]}_results_{dataset_name[i]}.json'
                )
        if strong_pseudo_labeling[i]:
            database['datasets'][pseudo_labelled_dataset_name[i]] = base.pseudo_label(
                database['datasets'][dataset_name[i]], event_classes,
                False, False, strong_pseudo_labeling[i],
                None, None, events[0],
            )
            with (storage_dir / f'{dataset_name[i]}_pseudo_labeled.tsv').open('w') as fid:
                fid.write('filename\tonset\toffset\tevent_label\n')
                for key, event_list in events[0].items():
                    if len(event_list) == 0:
                        fid.write(f'{key}.wav\t\t\t\n')
                    for t_on, t_off, event_label in event_list:
                        fid.write(
                            f'{key}.wav\t{t_on}\t{t_off}\t{event_label}\n')

    if any(strong_pseudo_labeling):
        dump_json(
            database,
            storage_dir / Path(data_provider.json_path).name,
            create_path=True,
            indent=4,
            ensure_ascii=False,
        )
    inference_dir = Path(strong_label_crnn_hyper_params_dir) / 'inference'
    os.makedirs(str(inference_dir), exist_ok=True)
    (inference_dir / storage_dir.name).symlink_to(storage_dir)
    emissions_tracker.stop()
    print(storage_dir)
