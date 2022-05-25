from copy import deepcopy
from pathlib import Path
import numpy as np

from paderbox.io.json_module import dump_json
from pb_sed.filters import medfilt
from pb_sed.models.base.inference import boundariesfilt
from sed_scores_eval import io, clip_based, intersection_based, collar_based
from sed_scores_eval.utils.auc import staircase_auc
from sed_scores_eval.utils.scores import validate_score_dataframe


def update_leaderboard(
        leaderboard, metric_name, metric_values, hyper_params_and_other_values,
        scores, minimize=False,
):
    if metric_name not in leaderboard:
        metric_values = {
            event_class: metric_values[event_class]
            for event_class in hyper_params_and_other_values
        }
        leaderboard[metric_name] = (
            metric_values, deepcopy(hyper_params_and_other_values), deepcopy(scores)
        )
    else:
        minimize = (
            minimize[metric_name] if isinstance(minimize, dict)
            else (metric_name in minimize)
            if isinstance(minimize, (list, tuple))
            else minimize
        )
        for event_class in hyper_params_and_other_values.keys():
            metric_value = metric_values[event_class]
            if (
                (metric_value * (-1)**minimize)
                    >= (leaderboard[metric_name][0][event_class] * (-1)**minimize)
            ):
                leaderboard[metric_name][0][event_class] = metric_value
                leaderboard[metric_name][1][event_class].update(
                    hyper_params_and_other_values[event_class])
                for audio_id in leaderboard[metric_name][2].keys():
                    leaderboard[metric_name][2][audio_id][event_class] = scores[audio_id][event_class]
    leaderboard[metric_name][0]['macro_average'] = float(np.mean([
        leaderboard[metric_name][0][event_class]
        for event_class in hyper_params_and_other_values.keys()
    ]))
    return leaderboard


def tune_tagging(
        tagging_scores, medfilt_length_candidates, metrics, minimize=False,
        storage_dir=None,
):
    leaderboard = {}
    audio_ids = sorted(tagging_scores.keys())
    event_classes = None
    for medfilt_len in medfilt_length_candidates:
        if medfilt_len > 1:
            scores_filtered = deepcopy(tagging_scores)
            for audio_id in audio_ids:
                timestamps, event_classes = validate_score_dataframe(
                    tagging_scores[audio_id], event_classes=event_classes
                )
                scores = tagging_scores[audio_id][event_classes].to_numpy()
                scores_filtered[audio_id][event_classes] = medfilt(
                    scores, medfilt_len, axis=0)
        else:
            scores_filtered = tagging_scores
        for metric_name, metric_fn in metrics.items():
            metric_values, other_values = metric_fn(scores_filtered)
            print()
            print(f'{metric_name}(medfilt_length={medfilt_len})')
            print(metric_values)
            hyper_params_and_other_values = {}
            for event_class in metric_values:
                if event_class.endswith('_average'):
                    continue
                hyper_params_and_other_values[event_class] = {
                    'medfilt_length': medfilt_len, **other_values.get(event_class, {})}
            leaderboard = update_leaderboard(
                leaderboard, metric_name, metric_values,
                hyper_params_and_other_values, scores_filtered,
                minimize=minimize
            )
    if storage_dir is not None:
        for metric_name in leaderboard:
            metric_values = leaderboard[metric_name][0]
            hyper_params_and_other_values = leaderboard[metric_name][1]
            for event_class in hyper_params_and_other_values:
                hyper_params_and_other_values[event_class][metric_name] = metric_values[event_class]
            dump_json(
                hyper_params_and_other_values,
                Path(storage_dir) / f'tagging_hyper_params_{metric_name}.json'
            )
    print()
    print('best:')
    for metric_name in metrics:
        print()
        print(metric_name, leaderboard[metric_name][0])
    return leaderboard


def boundaries_from_events(ground_truth):
    if isinstance(ground_truth, (str, Path)):
        ground_truth = io.read_ground_truth_events(ground_truth)
    boundaries_ground_truth = {}
    for audio_id, event_list in ground_truth.items():
        boundaries_ground_truth[audio_id] = {}
        for onset, offset, event_label in event_list:
            if event_label in boundaries_ground_truth[audio_id]:
                boundaries_ground_truth[audio_id][event_label] = (
                    boundaries_ground_truth[audio_id][event_label][0],
                    offset
                )
            else:
                boundaries_ground_truth[audio_id][event_label] = (onset, offset)
        boundaries_ground_truth[audio_id] = [
            (onset, offset, event_label)
            for event_label, (onset, offset)
            in boundaries_ground_truth[audio_id].items()
        ]
    return boundaries_ground_truth


def tune_boundaries_detection(
        detection_scores, medfilt_length_candidates,
        stepfilt_length_candidates, tags, metrics, minimize=False,
        tag_masking=None, storage_dir=None,
):
    if tag_masking in [True, False, '?']:
        tag_masking = {key: tag_masking for key in metrics.keys()}
    assert isinstance(tag_masking, dict), tag_masking
    assert tag_masking.keys() == metrics.keys(), (
        tag_masking.keys(), metrics.keys())
    assert all([val in [True, False, '?'] for val in tag_masking.values()])
    leaderboard = {}
    audio_ids = sorted(detection_scores.keys())
    event_classes = None
    for medfilt_len in medfilt_length_candidates:
        if medfilt_len > 1:
            scores_medfiltered = deepcopy(detection_scores)
            for audio_id in audio_ids:
                timestamps, event_classes = validate_score_dataframe(
                    detection_scores[audio_id], event_classes=event_classes
                )
                scores = detection_scores[audio_id][event_classes].to_numpy()
                scores_medfiltered[audio_id][event_classes] = medfilt(
                    scores, medfilt_len, axis=0)
        else:
            scores_medfiltered = detection_scores
        for stepfilt_len in stepfilt_length_candidates:
            scores_boundariesfiltered = deepcopy(scores_medfiltered)
            for audio_id in audio_ids:
                timestamps, event_classes = validate_score_dataframe(
                    scores_medfiltered[audio_id], event_classes=event_classes
                )
                scores = scores_medfiltered[audio_id][event_classes].to_numpy()
                scores_boundariesfiltered[audio_id][event_classes] = boundariesfilt(
                    scores, stepfilt_len, axis=0)
            scores_masked = deepcopy(scores_boundariesfiltered)
            for audio_id in audio_ids:
                timestamps, event_classes = validate_score_dataframe(
                    scores_masked[audio_id], event_classes=event_classes
                )
                scores_masked[audio_id][event_classes] *= tags[audio_id]
            for metric_name, metric_fn in metrics.items():
                if tag_masking[metric_name] == '?':
                    tag_masking_candidates = [False, True]
                else:
                    tag_masking_candidates = [tag_masking[metric_name]]
                for tag_masked in tag_masking_candidates:
                    scores = scores_masked if tag_masked else scores_boundariesfiltered
                    metric_values, other_values = metric_fn(scores)
                    print()
                    print(f'{metric_name}(medfilt_length={medfilt_len},stepfilt_length={stepfilt_len},tag_masked={tag_masked}):')
                    print(metric_values)
                    hyper_params_and_other_values = {}
                    for event_class in metric_values:
                        if event_class.endswith('_average'):
                            continue
                        hyper_params_and_other_values[event_class] = {
                            'medfilt_length': medfilt_len,
                            'stepfilt_length': stepfilt_len,
                            'tag_masked': tag_masked,
                            **other_values.get(event_class, {})
                        }
                    leaderboard = update_leaderboard(
                        leaderboard, metric_name, metric_values, hyper_params_and_other_values, scores,
                        minimize=minimize,
                    )
    if storage_dir is not None:
        for metric_name in leaderboard:
            metric_values = leaderboard[metric_name][0]
            hyper_params_and_other_values = leaderboard[metric_name][1]
            for event_class in hyper_params_and_other_values:
                hyper_params_and_other_values[event_class][metric_name] = metric_values[event_class]
            dump_json(
                hyper_params_and_other_values,
                Path(storage_dir) / f'boundaries_detection_hyper_params_{metric_name}.json'
            )
    print()
    print('best:')
    for metric_name in metrics:
        print()
        print(metric_name, ':')
        print(leaderboard[metric_name][0])
    return leaderboard


def tune_sound_event_detection(
        detection_scores, medfilt_length_candidates, tags, metrics,
        minimize=False, tag_masking=None, storage_dir=None,
):
    if tag_masking in [True, False, '?']:
        tag_masking = {key: tag_masking for key in metrics.keys()}
    assert isinstance(tag_masking, dict), tag_masking
    assert tag_masking.keys() == metrics.keys(), (
        tag_masking.keys(), metrics.keys())
    assert all([val in [True, False, '?'] for val in tag_masking.values()])
    leaderboard = {}
    audio_ids = sorted(detection_scores.keys())
    event_classes = None
    for medfilt_len in medfilt_length_candidates:
        if medfilt_len > 1:
            scores_filtered = deepcopy(detection_scores)
            for audio_id in audio_ids:
                timestamps, event_classes = validate_score_dataframe(
                    detection_scores[audio_id], event_classes=event_classes
                )
                scores = detection_scores[audio_id][event_classes].to_numpy()
                scores_filtered[audio_id][event_classes] = medfilt(
                    scores, medfilt_len, axis=0)
        else:
            scores_filtered = detection_scores
        scores_masked = deepcopy(scores_filtered)
        for audio_id in audio_ids:
            timestamps, event_classes = validate_score_dataframe(
                scores_masked[audio_id], event_classes=event_classes
            )
            scores_masked[audio_id][event_classes] *= tags[audio_id]
        for metric_name, metric_fn in metrics.items():
            if tag_masking[metric_name] == '?':
                tag_masking_candidates = [False, True]
            else:
                tag_masking_candidates = [tag_masking[metric_name]]
            for tag_masked in tag_masking_candidates:
                scores = scores_masked if tag_masked else scores_filtered
                metric_values, other_values = metric_fn(scores)
                print()
                print(f'{metric_name}(medfilt_length={medfilt_len},tag_masked={tag_masked}):')
                print(metric_values)
                hyper_params_and_other_values = {}
                for event_class in metric_values:
                    if event_class.endswith('_average'):
                        continue
                    hyper_params_and_other_values[event_class] = {
                        'medfilt_length': medfilt_len,
                        'tag_masked': tag_masked,
                        **other_values.get(event_class, {})
                    }
                leaderboard = update_leaderboard(
                    leaderboard, metric_name, metric_values, hyper_params_and_other_values, scores,
                    minimize=minimize
                )
    for metric_name in leaderboard:
        metric_values = leaderboard[metric_name][0]
        hyper_params_and_other_values = leaderboard[metric_name][1]
        for event_class in hyper_params_and_other_values:
            hyper_params_and_other_values[event_class][metric_name] = metric_values[event_class]
        if storage_dir is not None:
            dump_json(
                hyper_params_and_other_values,
                Path(storage_dir) / f'sed_hyper_params_{metric_name}.json'
            )
    print()
    print('best:')
    for metric_name in metrics:
        print()
        print(metric_name, ':')
        print(leaderboard[metric_name][0])
    return leaderboard


def f_tag(tagging_scores, *, ground_truth, num_jobs):
    best_f, best_p, best_r, thresholds, stats = clip_based.best_fscore(
        tagging_scores, ground_truth, num_jobs=num_jobs
    )
    return best_f, {
        key: {'threshold': thresholds[key]} for key in thresholds
    }


def f_collar(
        detection_scores, *, ground_truth,
        onset_collar, offset_collar, offset_collar_rate,
        min_precision=0., min_recall=0.,
        return_onset_offset_bias=False,
        num_jobs=1,
):
    best_f, best_p, best_r, thresholds, stats = collar_based.best_fscore(
        detection_scores, ground_truth, onset_collar=onset_collar,
        offset_collar=offset_collar, offset_collar_rate=offset_collar_rate,
        min_precision=min_precision, min_recall=min_recall,
        num_jobs=num_jobs
    )
    if return_onset_offset_bias:
        f, p, r, stats = collar_based.fscore(
            detection_scores, ground_truth, thresholds,
            onset_collar=onset_collar, offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
            return_onset_offset_dist_sum=True, num_jobs=num_jobs
        )
        return best_f, {
            key: {
                'threshold': thresholds[key],
                'onset_bias': stats[key]['onset_dist_sum'] / max(stats[key]['tps'], 1),
                'offset_bias': stats[key]['offset_dist_sum'] / max(stats[key]['tps'], 1),
            }
            for key in thresholds
        }
    return best_f, {
        key: {'threshold': thresholds[key]} for key in thresholds
    }


def psd_auc(
        detection_scores, *, ground_truth, audio_durations,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, num_jobs=1,
):
    psds, psdroc, classwise_rocs = intersection_based.psds(
        detection_scores, ground_truth, audio_durations,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr,
        time_decimals=time_decimals, num_jobs=num_jobs
    )
    aucs = {
        event_class: staircase_auc(tpr, efpr, max_efpr)
        for event_class, (tpr, efpr, *_) in classwise_rocs.items()
    }
    return aucs, {}
