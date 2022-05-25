

def add_label_types(example):
    if 'events_start_samples' in example or 'events_stop_samples' in example:
        # strong labels
        assert (
            'events' in example
            and 'events_start_samples' in example
            and 'events_stop_samples' in example
        ), example.keys()
        if 'label_types' not in example:
            example['label_types'] = len(example['events']) * ['strong']
        if 'unlabeled' not in example:
            example['unlabeled'] = False
    elif 'events' in example:
        # weak labels
        example['events_start_samples'] = [0 for _ in example['events']]
        example['events_stop_samples'] = [
            example['audio_data'].shape[-1] for _ in example['events']]
        if 'label_types' not in example:
            example['label_types'] = len(example['events']) * ['weak']
        if 'unlabeled' not in example:
            example['unlabeled'] = False
    else:
        # no labels
        example['events'] = []
        example['events_start_samples'] = []
        example['events_stop_samples'] = []
        example['label_types'] = []
        example['unlabeled'] = True
    return example
