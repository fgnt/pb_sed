
from padertorch.data.segment import Segmenter
import numpy as np
from math import ceil


def segment_batch(batch, max_length, overlap, keys=('stft',), axis=2):
    """

    Args:
        batch:
        max_length:
        overlap:
        keys:
        axis:

    Returns:

    >>> batch = {'example_id': ['a', 'b', 'c'], 'stft': np.cumsum(np.ones((3,1,50,7,2)), 2), 'seq_len':[50,47,46]}
    >>> segments = segment_batch(batch,12,2)
    >>> segments
    >>> len(segments), segments[0].keys(), segments[0]['stft'].shape
    """
    sequence_lengths_batch = batch["seq_len"]
    if max(sequence_lengths_batch) > max_length:
        segmenter = Segmenter(
            length=max_length, shift=max_length-overlap,
            include_keys=keys, copy_keys=('example_id',), axis=axis,
            mode='constant', padding=True,
        )
        segments = segmenter(batch)
        m = len(segments)
        print(
            f'Split batch with sequence length {max(sequence_lengths_batch)} '
            f'into {m} segments.')
        for i, segment in enumerate(segments):
            segment['example_id'] = [
                f'{example_id}_!segment!_{i}_{m}'
                for example_id in segment['example_id']
            ]
            segment['seq_len'] = [
                min(max_length, sl - segment['segment_start'])
                for sl in sequence_lengths_batch
            ]
            segment['stft'] = segment['stft'][:, :, :max(segment['seq_len'])]
    else:
        segments = [batch]
    return segments


def merge_segments(segmental_output, segment_overlap):
    merged_output = {}
    for audio_id in sorted(segmental_output.keys()):
        if "_!segment!_0_" in audio_id:
            audio_id, n_segments = audio_id.split("_!segment!_0_")
            n_segments = int(n_segments)
            merged_output[audio_id] = []
            for i in range(n_segments):
                score_arr = segmental_output[
                    f'{audio_id}_!segment!_{i}_{n_segments}']
                if i < (n_segments - 1) and segment_overlap > 0:
                    score_arr = score_arr[..., :-ceil(segment_overlap/2), :]
                if i > 0 and segment_overlap > 0:
                    score_arr = score_arr[..., segment_overlap//2:, :]
                merged_output[audio_id].append(score_arr)
            merged_output[audio_id] = np.concatenate(
                merged_output[audio_id], axis=-2
            )
        elif "_!segment!_" not in audio_id:
            merged_output[audio_id] = segmental_output[audio_id]
    return merged_output
