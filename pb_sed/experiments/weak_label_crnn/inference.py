import os
import numpy as np
from pathlib import Path
from copy import deepcopy
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from codecarbon import EmissionsTracker

from paderbox.utils.timer import timeStamped
from paderbox.io.json_module import load_json, dump_json
from sed_scores_eval import clip_based, intersection_based, collar_based
from sed_scores_eval.base_modules.detection import scores_to_event_list
from sed_scores_eval.utils.auc import staircase_auc
from sed_scores_eval import io

from pb_sed.paths import storage_root
from pb_sed.models.weak_label import CRNN
from pb_sed.models import base
from pb_sed.data_preparation.provider import DataProvider
from pb_sed.utils.segment import merge_segments

ex_name = 'weak_label_crnn_inference'
ex = Exp(ex_name)


@ex.config
def config():
    debug = False
    timestamp = timeStamped('')[1:] + ('_debug' if debug else '')

    hyper_params_dir = ''
    assert len(hyper_params_dir) > 0, 'Set hyper_params_dir on the command line.'

    tuning_config = load_json(Path(hyper_params_dir) / '1' / 'config.json')
    crnn_dirs = tuning_config['crnn_dirs']
    assert len(crnn_dirs) > 0, 'crnn_dirs must not be empty.'
    crnn_checkpoints = tuning_config['crnn_checkpoints']
    data_provider = tuning_config['data_provider']
    database_name = tuning_config['database_name']
    storage_dir = str(storage_root / 'weak_label_crnn' / database_name / 'inference' / timestamp)
    assert not Path(storage_dir).exists()
    del tuning_config
    sed_hyper_params_name = ['f', 'psds1']

    device = 0

    dataset_name = 'eval_public'
    ground_truth_filepath = None

    max_segment_length = None
    segment_overlap = 0
    save_scores = False
    save_detections = False

    weak_pseudo_labeling = False
    boundary_pseudo_labeling = False
    strong_pseudo_labeling = False
    pseudo_labeled_dataset_name = dataset_name

    pseudo_widening = .0

    ex.observers.append(FileStorageObserver.create(storage_dir))


def tagging(
        crnns, dataset, device, timestamps, event_classes, hyper_params_dir,
        ground_truth, audio_durations, psds_params=(),
        max_segment_length=None, segment_overlap=None,
):
    print()
    print('Tagging')
    hyper_params = load_json(Path(hyper_params_dir) / 'tagging_hyper_params_f.json')
    thresholds = {
        event_class: hyper_params[event_class]['threshold']
        for event_class in hyper_params
    }
    tagging_scores = base.tagging(
        crnns, dataset, device,
        max_segment_length=max_segment_length,
        segment_overlap=segment_overlap,
        merge_score_segments=False,
    )
    results = {}
    if ground_truth is not None:
        tagging_scores_merged = merge_segments(
            tagging_scores, segment_overlap=0)
        tagging_scores_df = base.scores_to_dataframes(
            tagging_scores_merged,
            timestamps=timestamps, event_classes=event_classes,
        )
        if ground_truth:
            f, p, r, stats = clip_based.fscore(
                tagging_scores_df, ground_truth, thresholds, num_jobs=8
            )
            print('f', f)
            print('p', p)
            print('r', r)
            for key in f:
                results.update({
                    f'{key}_f': f[key],
                    f'{key}_p': p[key],
                    f'{key}_r': r[key],
                })
            for j in range(len(psds_params)):
                psds, psd_roc, classwise_rocs = intersection_based.psds(
                    tagging_scores_df, ground_truth, audio_durations,
                    **psds_params[j], num_jobs=8,
                )
                print(f'psds[{j}]', psds)
                results[f'psds[{j}]'] = psds
                for event_class, (tpr, efpr, *_) in classwise_rocs.items():
                    results[f'{event_class}_auc[{j}]'] = staircase_auc(
                        tpr, efpr, psds_params[j].get('max_efpr', 100))
                psds, psd_roc, classwise_rocs = intersection_based.reference.approximate_psds(
                    tagging_scores_df, ground_truth, audio_durations,
                    **psds_params[j], thresholds=np.linspace(.01, .99, 50),
                )
                print(f'approx_psds[{j}]', psds)
                results[f'approx_psds[{j}]'] = psds
                for event_class, (tpr, efpr, *_) in classwise_rocs.items():
                    results[f'{event_class}_approx_auc[{j}]'] = staircase_auc(
                        tpr, efpr, psds_params[j].get('max_efpr', 100))

    thresholds = np.array([
        thresholds[event_class] for event_class in event_classes])
    tagging_scores = {
        audio_id: tagging_scores[audio_id][0]
        for audio_id in tagging_scores.keys()
    }
    tags = {
        audio_id: tagging_scores[audio_id] > thresholds
        for audio_id in tagging_scores.keys()
    }
    return tags, tagging_scores, results


def boundaries_detection(
        crnns, dataset, device, timestamps, event_classes, tags,
        hyper_params_dir, ground_truth, collar_based_params,
        max_segment_length=None, segment_overlap=None, pseudo_widening=.0,
):
    print()
    print('Boundaries Detection')
    hyper_params = load_json(Path(hyper_params_dir) / f'boundaries_detection_hyper_params_f.json')
    stepfilt_length = np.array([
        hyper_params[event_class]['stepfilt_length']
        for event_class in event_classes])
    thresholds = {
        event_class: hyper_params[event_class]['threshold']
        for event_class in event_classes
    }
    boundary_scores = base.boundaries_detection(
        crnns, dataset, device,
        stepfilt_length=stepfilt_length,
        apply_mask=True, masks=tags,
        max_segment_length=max_segment_length,
        segment_overlap=segment_overlap,
        merge_score_segments=True,
        timestamps=timestamps,
        event_classes=event_classes,
    )
    results = {}
    if ground_truth:
        boundary_ground_truth = base.tuning.boundaries_from_events(ground_truth)
        f, p, r, stats = collar_based.fscore(
            boundary_scores, boundary_ground_truth, thresholds,
            **collar_based_params,
            return_onset_offset_dist_sum=True, num_jobs=8,
        )
        print('f', f)
        print('p', p)
        print('r', r)
        for key in f:
            results.update({
                f'{key}_f': f[key],
                f'{key}_p': p[key],
                f'{key}_r': r[key],
            })
            if key in stats:
                results.update({
                    f'{key}_onset_bias': stats[key]['onset_dist_sum'] / max(stats[key]['tps'], 1),
                    f'{key}_offset_bias': stats[key]['offset_dist_sum'] / max(stats[key]['tps'], 1),
                })

    boundaries_detection = scores_to_event_list(
        boundary_scores, thresholds, event_classes=event_classes)

    for clip_id in boundaries_detection:
        boundaries_in_clip = []
        for onset, offset, event_label in boundaries_detection[clip_id]:
            onset = max(np.round(onset - pseudo_widening - hyper_params[event_label]['onset_bias'], 3), 0)
            offset = np.round(offset + pseudo_widening - hyper_params[event_label]['offset_bias'], 3)
            if offset > onset:
                boundaries_in_clip.append((onset, offset, event_label))
        boundaries_detection[clip_id] = boundaries_in_clip
    return boundaries_detection, results


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

    window_lengths = np.zeros((len(hyper_params), len(event_classes)))
    medfilt_lengths = np.zeros((len(hyper_params), len(event_classes)))
    tag_masked = np.zeros((len(hyper_params), len(event_classes)))
    window_shift = set()
    for i, hyper_params_i in enumerate(hyper_params):
        for j, event_class in enumerate(event_classes):
            window_lengths[i, j] = hyper_params_i[event_class]['window_length']
            medfilt_lengths[i, j] = hyper_params_i[event_class]['medfilt_length']
            tag_masked[i, j] = hyper_params_i[event_class]['tag_masked']
            window_shift.add(hyper_params_i[event_class]['window_shift'])
    if not len(window_shift) == 1:
        raise ValueError(
            'Inference with multiple window shifts is not supported.'
        )
    window_shift = list(window_shift)[0]
    if max_segment_length is not None:
        assert max_segment_length % window_shift == 0, (max_segment_length, window_shift)
        assert (segment_overlap//2) % window_shift == 0, (segment_overlap, window_shift)
    detection_scores = base.sound_event_detection(
        crnns, dataset, device,
        model_kwargs={
            'window_length': window_lengths, 'window_shift': window_shift},
        medfilt_length=medfilt_lengths, apply_mask=tag_masked, masks=tags,
        timestamps=timestamps[::window_shift], event_classes=event_classes,
        max_segment_length=max_segment_length, segment_overlap=segment_overlap,
        merge_score_segments=True, score_segment_overlap=segment_overlap//window_shift,
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
                    detection_scores[i], thresholds, event_classes=event_classes,
                )
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
        storage_dir, hyper_params_dir, sed_hyper_params_name,
        crnn_dirs, crnn_checkpoints, device,
        data_provider, dataset_name, ground_truth_filepath,
        save_scores, save_detections, max_segment_length, segment_overlap,
        weak_pseudo_labeling, boundary_pseudo_labeling, strong_pseudo_labeling,
        pseudo_widening, pseudo_labeled_dataset_name,
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

    boundary_collar_based_params = {
        'onset_collar': .5,
        'offset_collar': .5,
        'offset_collar_rate': .0,
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
        CRNN.from_storage_dir(
            storage_dir=crnn_dir, config_name='1/config.json',
            checkpoint_name=crnn_checkpoint
        )
        for crnn_dir, crnn_checkpoint in zip(crnn_dirs, crnn_checkpoints)
    ]
    print('Params', sum([p.numel() for crnn in crnns for p in crnn.parameters()]))
    print('CNN2d Params', sum([p.numel() for crnn in crnns for p in crnn.cnn.cnn_2d.parameters()]))
    data_provider = DataProvider.from_config(data_provider)
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
    if not isinstance(weak_pseudo_labeling, list):
        weak_pseudo_labeling = len(dataset_name)*[weak_pseudo_labeling]
    assert len(weak_pseudo_labeling) == len(dataset_name)
    if not isinstance(boundary_pseudo_labeling, list):
        boundary_pseudo_labeling = len(dataset_name)*[boundary_pseudo_labeling]
    assert len(boundary_pseudo_labeling) == len(dataset_name)
    if not isinstance(strong_pseudo_labeling, list):
        strong_pseudo_labeling = len(dataset_name)*[strong_pseudo_labeling]
    assert len(strong_pseudo_labeling) == len(dataset_name)
    if not isinstance(pseudo_labeled_dataset_name, list):
        pseudo_labeled_dataset_name = [pseudo_labeled_dataset_name]
    assert len(pseudo_labeled_dataset_name) == len(dataset_name)

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
                    0, audio_durations[audio_id],
                    (max_segment_length-segment_overlap)*frame_shift
                )
                timestamps[audio_id] = np.concatenate((
                    ts, [audio_durations[audio_id]]
                ))
        tags, tagging_scores, tagging_results = tagging(
            crnns, dataset, device, timestamps, event_classes,
            hyper_params_dir, ground_truth_filepath[i], audio_durations,
            [psds_scenario_1, psds_scenario_2],
            max_segment_length=max_segment_length,
            segment_overlap=segment_overlap,
        )
        if tagging_results:
            dump_json(
                tagging_results,
                storage_dir / f'tagging_results_{dataset_name[i]}.json'
            )

        timestamps = np.round(np.arange(0, 100000) * frame_shift, decimals=6)
        if ground_truth_filepath[i] is not None or boundary_pseudo_labeling[i]:
            boundaries, boundaries_detection_results = boundaries_detection(
                crnns, dataset, device, timestamps, event_classes, tags,
                hyper_params_dir, ground_truth_filepath[i],
                boundary_collar_based_params,
                max_segment_length=max_segment_length,
                segment_overlap=segment_overlap,
                pseudo_widening=pseudo_widening,
            )
            if boundaries_detection_results:
                dump_json(
                    boundaries_detection_results,
                    storage_dir / f'boundaries_detection_results_{dataset_name[i]}.json'
                )
        else:
            boundaries = {}
        if not isinstance(sed_hyper_params_name, (list, tuple)):
            sed_hyper_params_name = [sed_hyper_params_name]
        if (ground_truth_filepath[i] is not None) or strong_pseudo_labeling[i] or save_scores or save_detections:
            events, sed_results = sound_event_detection(
                crnns, dataset, device, timestamps, event_classes, tags,
                hyper_params_dir, sed_hyper_params_name,
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
        else:
            events = [{}]
        database['datasets'][pseudo_labeled_dataset_name[i]] = base.pseudo_label(
            database['datasets'][dataset_name[i]], event_classes,
            weak_pseudo_labeling[i],
            boundary_pseudo_labeling[i],
            strong_pseudo_labeling[i],
            tags, boundaries, events[0],
        )

    if any(weak_pseudo_labeling) or any(boundary_pseudo_labeling) or any(strong_pseudo_labeling):
        dump_json(
            database,
            storage_dir / Path(data_provider.json_path).name,
            create_path=True,
            indent=4,
            ensure_ascii=False,
        )
    inference_dir = Path(hyper_params_dir) / 'inference'
    os.makedirs(str(inference_dir), exist_ok=True)
    (inference_dir / storage_dir.name).symlink_to(storage_dir)
    emissions_tracker.stop()
    print(storage_dir)
