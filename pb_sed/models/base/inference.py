import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from pb_sed.filters import medfilt, stepfilt
from pb_sed.utils.segment import segment_batch, merge_segments
from sed_scores_eval.utils.scores import create_score_dataframe
from sed_scores_eval import io
from padertorch.ops.sequence.mask import compute_mask


def tagging(
        models, dataset, device,
        max_segment_length=None, segment_overlap=None,
        merge_score_segments=False, score_segment_overlap=None,
        model_kwargs=None, medfilt_length=1, method='tagging',
        timestamps=None, event_classes=None, score_storage_dir=None,
):
    def post_processing_fn(x):
        return x.max(-2, keepdims=True)

    return inference(
        models, method, dataset, device,
        max_segment_length=max_segment_length,
        segment_overlap=segment_overlap,
        merge_score_segments=merge_score_segments,
        score_segment_overlap=score_segment_overlap,
        model_kwargs=model_kwargs,
        medfilt_length=medfilt_length,
        post_processing_fn=post_processing_fn,
        timestamps=timestamps,
        event_classes=event_classes,
        score_storage_dir=score_storage_dir,
    )


def boundaries_detection(
        models, dataset, device,
        max_segment_length=None, segment_overlap=None,
        merge_score_segments=False, score_segment_overlap=None,
        model_kwargs=None, medfilt_length=1, stepfilt_length=0,
        apply_mask=False, masks=None, method='boundaries_detection',
        timestamps=None, event_classes=None, score_storage_dir=None,
):
    return inference(
        models, method, dataset, device,
        max_segment_length=max_segment_length,
        segment_overlap=segment_overlap,
        merge_score_segments=merge_score_segments,
        score_segment_overlap=score_segment_overlap,
        model_kwargs=model_kwargs,
        medfilt_length=medfilt_length,
        stepfilt_length=stepfilt_length,
        apply_mask=apply_mask,
        masks=masks,
        timestamps=timestamps,
        event_classes=event_classes,
        score_storage_dir=score_storage_dir,
    )


def sound_event_detection(
        models, dataset, device,
        max_segment_length=None, segment_overlap=None,
        merge_score_segments=False, score_segment_overlap=None,
        model_kwargs=None, medfilt_length=1, method='sound_event_detection',
        apply_mask=False, masks=None,
        timestamps=None, event_classes=None, score_storage_dir=None,
):
    return inference(
        models, method, dataset, device,
        max_segment_length=max_segment_length,
        segment_overlap=segment_overlap,
        merge_score_segments=merge_score_segments,
        score_segment_overlap=score_segment_overlap,
        model_kwargs=model_kwargs,
        medfilt_length=medfilt_length,
        apply_mask=apply_mask,
        masks=masks,
        timestamps=timestamps,
        event_classes=event_classes,
        score_storage_dir=score_storage_dir,
    )


def inference(
        model, method, dataset, device,
        max_segment_length=None, segment_overlap=0,
        merge_score_segments=False, score_segment_overlap=None,
        model_kwargs=None, medfilt_length=1, stepfilt_length=None,
        apply_mask=False, masks=None, post_processing_fn=None,
        timestamps=None, event_classes=None, score_storage_dir=None,
):
    if not isinstance(model, (list, tuple)):
        model = [model]
    if model_kwargs is None:
        model_kwargs = {}
    if not isinstance(model_kwargs, (list, tuple)):
        model_kwargs = len(model)*[model_kwargs]
    else:
        assert len(model_kwargs) == len(model), (len(model), len(model_kwargs))

    medfilt_length = np.array(medfilt_length, dtype=np.int)
    apply_mask = np.array(apply_mask, dtype=np.bool)

    for i in range(len(model)):
        assert hasattr(model[i], method), (model[i], method)
        model[i].to(device)
        model[i].eval()

    scores = {}
    with torch.no_grad():
        score_cache = {}
        for batch in tqdm(dataset):
            if 'weak_targets' in batch:
                batch.pop('weak_targets')
            if 'boundary_targets' in batch:
                batch.pop('boundary_targets')
            if 'strong_targets' in batch:
                batch.pop('strong_targets')
            if max_segment_length is not None:
                input_segments = segment_batch(
                    batch,
                    max_length=max_segment_length,
                    overlap=segment_overlap
                )
            else:
                input_segments = [batch]
            for segment in input_segments:
                segment = model[0].example_to_device(segment, device)
                segment_scores = []
                seq_len = None
                for i in range(len(model)):
                    yi, seq_len_i = getattr(model[i], method)(
                        segment, **model_kwargs[i])
                    segment_scores.append(yi.detach().cpu().numpy())
                    if i == 0:
                        seq_len = seq_len_i
                    else:
                        assert (seq_len_i == seq_len).all(), (
                            seq_len, seq_len_i)
                segment_scores = np.mean(segment_scores, axis=0)
                sequence_mask = compute_mask(
                    torch.from_numpy(segment_scores), seq_len,
                    batch_axis=0, sequence_axis=-1,
                ).numpy()
                segment_scores = segment_scores * sequence_mask
                # median filtering:
                segment_scores = filtering(
                    segment_scores, medfilt, medfilt_length)

                if stepfilt_length is not None:
                    # boundary filtering:
                    stepfilt_length = np.array(stepfilt_length, dtype=np.int)
                    segment_scores = filtering(
                        segment_scores, boundariesfilt, stepfilt_length)

                # separate examples within batch
                if post_processing_fn is None:
                    def post_processing_fn(x):
                        return x

                score_cache.update({
                    audio_id: post_processing_fn(
                        segment_scores[i, ..., :sl].swapaxes(-2, -1)
                    )
                    for i, (audio_id, sl) in enumerate(zip(
                        segment['example_id'], seq_len))
                })

                # applying mask allows to, e.g, mask SED score by tags.
                if apply_mask.any():
                    assert masks is not None
                    for audio_id in score_cache:
                        assert audio_id in masks, audio_id
                        if apply_mask.ndim == 2:
                            apply_mask = apply_mask[..., None, :]
                        # elif apply_mask.ndim > 2:
                        #     raise ValueError(
                        #         f'apply_mask must be 0-,1- or 2-dimensional '
                        #         f'but shape {apply_mask.shape} was given.'
                        #     )
                        mask = np.maximum(masks[audio_id], 1 - apply_mask)
                        score_cache[audio_id] *= mask
            if merge_score_segments:
                if '_!segment!_' in segment['example_id'][0]:
                    seg_idx, n_segments = segment['example_id'][0].split('_!segment!_')[-1].split('_')
                    seg_idx = int(seg_idx)
                    n_segments = int(n_segments)
                    if seg_idx == n_segments - 1:
                        score_cache = merge_segments(
                            score_cache,
                            segment_overlap=segment_overlap
                            if score_segment_overlap is None else score_segment_overlap
                        )
                    else:
                        continue
            if (
                timestamps is not None or event_classes is not None
                or score_storage_dir is not None
            ):
                assert timestamps is not None
                assert event_classes is not None
                score_cache = scores_to_dataframes(
                    score_cache, timestamps, event_classes, score_storage_dir,
                )
            if score_storage_dir is None:
                if not scores:
                    scores = score_cache
                elif isinstance(scores, (list, tuple)):
                    assert isinstance(score_cache, (list, tuple))
                    assert len(score_cache) == len(scores)
                    for i in range(len(scores)):
                        scores[i].update(score_cache[i])
                else:
                    assert isinstance(scores, dict)
                    assert isinstance(score_cache, dict)
                    scores.update(score_cache)
            else:
                scores = score_cache
            score_cache = {}
    return scores


def filtering(score_arr, filter_fn, filter_length):
    b, *_, k, t = score_arr.shape
    if filter_length.ndim == 0:
        score_arr = filter_fn(
            score_arr, filter_length, axis=-1)
    elif filter_length.ndim == 1:
        assert filter_length.shape[0] == k, filter_length.shape
        for cls_idx, filt_len in enumerate(filter_length):
            score_arr[..., cls_idx, :] = filter_fn(
                score_arr[..., cls_idx, :], filt_len, axis=-1)
    elif filter_length.ndim == 2:
        assert filter_length.shape[1] in [1, k], filter_length.shape
        n = filter_length.shape[0]
        if score_arr.ndim == 3:
            score_arr = np.broadcast_to(
                score_arr[:, None], (b, n, k, t)).copy()
        elif score_arr.ndim == 4:
            assert n == score_arr.shape[1], (score_arr.shape, n)
        else:
            raise ValueError(
                'scores returned by model must be 3- or 4-dimensional.'
            )
        for j in range(n):
            if filter_length.shape[1] == 1:
                score_arr[:, j] = filter_fn(
                    score_arr[:, j], filter_length[j, 0],
                    axis=-1
                )
            else:
                for cls_idx in range(k):
                    score_arr[:, j, cls_idx] = filter_fn(
                        score_arr[:, j, cls_idx], filter_length[j, cls_idx],
                        axis=-1,
                    )
    else:
        raise ValueError(
            f'filter_length.ndim must not be greater than 2 but '
            f'{filter_length} was given.')
    return score_arr


def boundariesfilt(score_arr, stepfilt_length, axis):
    if stepfilt_length > 0:
        temp_scores_fwd = stepfilt(
            score_arr, stepfilt_length, axis=axis
        )
        temp_scores_bwd = stepfilt(
            np.flip(score_arr, axis=axis), stepfilt_length, axis=axis
        )
    else:
        temp_scores_fwd = score_arr
        temp_scores_bwd = np.flip(score_arr, axis=axis)
    return np.minimum(
        torch.cummax(
            torch.from_numpy(temp_scores_fwd.copy()),
            dim=axis
        )[0].numpy(),
        np.flip(
            torch.cummax(
                torch.from_numpy(temp_scores_bwd.copy()),
                dim=axis
            )[0].numpy(),
            axis=axis
        ),
    )


def scores_to_dataframes(scores, timestamps, event_classes, storage_path=None):
    if isinstance(scores, np.ndarray):
        t, k = scores.shape
        assert len(timestamps) > t, (len(timestamps), t)
        assert len(event_classes) == k, (event_classes, k)
        scores = create_score_dataframe(
            scores, timestamps[:t+1], event_classes
        )
        if storage_path is not None:
            io.write_sed_scores(scores, storage_path)
        return scores

    assert isinstance(scores, dict), type(scores)
    audio_ids = sorted(scores.keys())
    score_dataframes = {}
    for audio_id in audio_ids:
        ts = (
            timestamps[audio_id] if isinstance(timestamps, dict)
            else timestamps
        )
        if scores[audio_ids[0]].ndim == 3:
            n = scores[audio_ids[0]].shape[0]
            for i in range(n):
                scores_i = scores[audio_id][i]
                if not score_dataframes:
                    score_dataframes = [{} for _ in range(n)]
                if storage_path is None:
                    filepath = None
                    score_dataframes[i][audio_id] = scores_to_dataframes(
                        scores_i, ts, event_classes, None)
                else:
                    assert isinstance(storage_path, (list, tuple)), type(storage_path)
                    assert len(storage_path) == n, (len(storage_path), n)

                    storage_path[i] = Path(storage_path[i])
                    storage_path[i].mkdir(parents=True, exist_ok=True)
                    filepath = storage_path[i] / f'{audio_id}.tsv'
                score_dataframes[i][audio_id] = scores_to_dataframes(
                    scores_i, ts, event_classes, filepath)
        else:
            if storage_path is None:
                filepath = None
            else:
                assert isinstance(storage_path, (str, Path)), (
                    type(storage_path))
                storage_path = Path(storage_path)
                storage_path.mkdir(parents=True, exist_ok=True)
                filepath = Path(storage_path) / f'{audio_id}.tsv'
            score_dataframes[audio_id] = scores_to_dataframes(
                scores[audio_id], ts, event_classes, filepath
            )

    if storage_path is None:
        return score_dataframes
    elif isinstance(storage_path, (list, tuple)):
        return [
            io.lazy_sed_scores_loader(p) for p in storage_path
        ]
    elif isinstance(storage_path, (str, Path)):
        return io.lazy_sed_scores_loader(storage_path)
    else:
        raise ValueError(
            f'storage_path must be of type None,str,Path,list or tuple '
            f'but type {type(storage_path)} was given.'
        )
