"""
This script prepares json files comprising information about the DESED
database. Information about the database can be found here:

https://github.com/turpaultn/DESED

For each audio clip a dict is stored containing the following keys:
audio_path: path to audio file
audio_length: length of audio clip in seconds

and if available:
events: list of events present in the audio clip
events_start_times: onset times in seconds for each event in events list
events_stop_times: offset times in seconds for each event in events list

Example usage:
python -m pb_sed.database.desed.create_json -db /path/to/desed
"""

import click
import pandas as pd
from copy import deepcopy
from natsort import natsorted
from pathlib import Path
from paderbox.io.json_module import dump_json
from pb_sed.paths import database_jsons_dir, pb_sed_root
from pb_sed.database.helper import prepare_sound_dataset
from sed_scores_eval import io


target_events = [
    'Alarm_bell_ringing',
    'Blender',
    'Cat',
    'Dishes',
    'Dog',
    'Electric_shaver_toothbrush',
    'Frying',
    'Running_water',
    'Speech',
    'Vacuum_cleaner',
]


def construct_json(database_path):
    """

    Args:
        database_path:

    Returns:

    """
    database = {
        'datasets': dict()
    }
    for purpose in ['train', 'validation', 'eval']:
        audio_base_dir = database_path / 'audio' / purpose
        for subdir in audio_base_dir.iterdir():
            name = subdir.name
            if name == purpose:
                dataset_name = purpose
            else:
                dataset_name = f'{purpose}_{name}'
            ground_truth_file = database_path / 'metadata' / purpose / f"{name}.tsv"
            audio_dir = audio_base_dir / name
            if ground_truth_file.exists() and name != 'unlabel_in_domain':
                ground_truth = read_ground_truth_file(ground_truth_file)
                clip_ids = ground_truth.keys()
            else:
                ground_truth = None
                clip_ids = [audio_file.name[:-len(".wav")] for audio_file in audio_dir.glob("*.wav")]
            examples = {}
            for clip_id in natsorted(clip_ids):
                audio_path = Path(audio_dir) / f'{clip_id}.wav'
                examples[clip_id] = {
                    'audio_path': str(audio_path),
                }
            if 'synthetic' in name or dataset_name in ['validation', 'eval_public', 'train_strong']:
                assert ground_truth is not None
                add_strong_labels(examples, ground_truth)
            elif ground_truth:
                assert dataset_name == 'train_weak', name
                add_weak_labels(examples, ground_truth)
            database['datasets'][dataset_name], missing = prepare_sound_dataset(examples)
            print(f'{len(missing)} from {len(clip_ids)} files missing in {dataset_name}')
            events = {
                event
                for example in database["datasets"][dataset_name].values()
                for event in example.get('events', [])
            }
            print(f'Number of event labels in {dataset_name}:', len(events))

    events = {
        event
        for ds in database['datasets'].values()
        for example in ds.values()
        for event in example.get('events', [])
    }
    print('Number of event labels:', len(events))
    return database


def read_ground_truth_file(filepath):
    file = pd.read_csv(filepath, sep='\t')
    if 'onset' in file.columns:
        # events
        return io.read_ground_truth_events(filepath)
    return io.read_ground_truth_tags(filepath)[0]


def add_strong_labels(examples, events):
    for clip_id in examples:
        event_list = events[clip_id]
        if len(event_list) > 0:
            assert isinstance(event_list[0], (list, tuple)), event_list
            event_list = [event for event in event_list if event[2] in target_events]
            for event in event_list:
                if event[2] not in target_events:
                    print(events[2])
            event_onsets, event_offsets, event_list = list(zip(*event_list))
        else:
            event_onsets, event_offsets, event_list = [], [], []
        examples[clip_id][f'events_start_times'] = event_onsets
        examples[clip_id][f'events_stop_times'] = event_offsets
        examples[clip_id]['events'] = event_list
    return examples


def add_weak_labels(examples, events):
    for clip_id in examples:
        event_list = events[clip_id]
        if len(event_list) > 0 and isinstance(event_list[0], (list, tuple)):
            event_list = [event for event in event_list if event[2] in target_events]
            event_onsets, event_offsets, event_list = list(zip(*event_list))
        examples[clip_id]['events'] = [
            event for event in event_list if event in target_events
        ]
    return examples


def create_jsons(database_path: Path, json_path: Path, indent=4):
    assert database_path.is_dir(), (
        f'Path "{str(database_path.absolute())}" is not a directory.'
    )
    database = construct_json(database_path)
    dump_json(
        database,
        json_path / 'desed.json',
        create_path=True,
        indent=indent,
        ensure_ascii=False,
    )
    print(f'Dumped json {json_path / "desed.json"}')
    database_pseudo_labeled = deepcopy(database)
    pseudo_labels_dir = pb_sed_root / 'exp' / 'strong_label_crnn_inference' / '2022-05-04-09-05-53'
    add_strong_labels(
        database_pseudo_labeled['datasets']['train_weak'],
        read_ground_truth_file(pseudo_labels_dir / 'train_weak_pseudo_labeled.tsv')
    )
    add_strong_labels(
        database_pseudo_labeled['datasets']['train_unlabel_in_domain'],
        read_ground_truth_file(pseudo_labels_dir / 'train_unlabel_in_domain_pseudo_labeled.tsv')
    )
    dump_json(
        database_pseudo_labeled,
        json_path / 'desed_pseudo_labeled_without_external.json',
        create_path=True,
        indent=indent,
        ensure_ascii=False,
        )
    print(f'Dumped json {json_path / "desed_pseudo_labeled_without_external.json"}')
    database_pseudo_labeled = deepcopy(database)
    pseudo_labels_dir = pb_sed_root / 'exp' / 'strong_label_crnn_inference' / '2022-06-24-10-06-21'
    add_strong_labels(
        database_pseudo_labeled['datasets']['train_weak'],
        read_ground_truth_file(pseudo_labels_dir / 'train_weak_pseudo_labeled.tsv')
    )
    add_strong_labels(
        database_pseudo_labeled['datasets']['train_unlabel_in_domain'],
        read_ground_truth_file(pseudo_labels_dir / 'train_unlabel_in_domain_pseudo_labeled.tsv')
    )
    dump_json(
        database_pseudo_labeled,
        json_path / 'desed_pseudo_labeled_with_external.json',
        create_path=True,
        indent=indent,
        ensure_ascii=False,
    )
    print(f'Dumped json {json_path / "desed_pseudo_labeled_with_external.json"}')


@click.command()
@click.option(
    '--database-path', '-db',
    help='Path where the database is located.',
    type=click.Path(),
)
@click.option(
    '--json-path', '-j',
    default=str(database_jsons_dir),
    help=f'Directory path where to save the generated JSON files. If a file '
         f'already exists, it gets overwritten. Defaults to "{database_jsons_dir}".',
    type=click.Path(),
)
def main(database_path, json_path):
    create_jsons(Path(database_path).absolute(), Path(json_path).absolute())


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
