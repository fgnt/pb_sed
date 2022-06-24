import numpy as np
from copy import deepcopy


def pseudo_label(
        dataset, event_classes,
        pseudo_tags, pseudo_boundaries, pseudo_events,
        tags, boundaries, events,
):
    if not any([pseudo_tags, pseudo_boundaries, pseudo_events]):
        return dataset
    dataset = deepcopy(dataset)
    assert not (pseudo_events and pseudo_boundaries)
    for audio_id in sorted(dataset.keys()):
        if pseudo_tags:
            dataset[audio_id]['events'] = sorted([
                event_class for value, event_class in zip(
                    tags[audio_id], event_classes
                ) if value > 0.5
            ])
        dataset[audio_id]['label_types'] = len(dataset[audio_id]['events']) * ['weak']
        if pseudo_events:
            set_onset_offset_times(
                dataset[audio_id], events[audio_id], "strong")
        elif pseudo_boundaries:
            set_onset_offset_times(
                dataset[audio_id], boundaries[audio_id], "boundaries")
    print()
    print(
        'label rate',
        np.mean([
            len(dataset[audio_id]['events']) > 0
            for audio_id in sorted(dataset.keys())
        ])
    )
    for label_type in ['weak', 'boundaries', 'strong']:
        print(
            f'pseudo {label_type} labels rate',
            np.mean([
                t == label_type
                for audio_id in sorted(dataset.keys())
                for t in dataset[audio_id]['label_types']
            ])
        )
    return dataset


def set_onset_offset_times(example, detections, label_type='strong'):
    event_labels = sorted({
        event_label for _, _, event_label in detections})
    assert "events" in example, example.keys()
    tags = sorted(set(example['events']))
    events = sorted(
        [
            event for event in detections if event[2] in tags
        ] + [
            (0., example['audio_length'], event_class)
            for event_class in tags if event_class not in event_labels
        ]
    )
    (
        example['events_start_times'],
        example['events_stop_times'],
        example['events'],
    ) = list(zip(*events)) if events else ([], [], [])
    example['label_types'] = [
        label_type if event in event_labels else "weak"
        for event in example['events']
    ]
