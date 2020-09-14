from natsort import natsorted
from pathlib import Path
from collections import defaultdict
import click
from paderbox.io.json_module import dump_json
from pb_sed.paths import database_jsons_dir
from pb_sed.database.helper import prepare_sound_dataset


def construct_json(database_path):
    database = {
        'datasets': dict()
    }

    # desed_real
    real_dataset_path = database_path/'real'
    for purpose in ['train', 'validation', 'eval']:
        files = sorted((real_dataset_path / 'metadata' / purpose).glob('*.tsv'))
        for segment_file in files:
            name = segment_file.name[:-len('.tsv')]
            audio_dir = real_dataset_path / 'audio' / purpose
            if '_pseudo_' in name:
                audio_dir = audio_dir / name.split('_pseudo_')[0]
            else:
                audio_dir = audio_dir / name
            segment_ids = read_segments_file(segment_file)
            examples = {}
            for segment_id in natsorted(segment_ids):
                audio_path = Path(audio_dir)/f'{segment_id}.wav'
                examples[segment_id] = {
                    'audio_path': str(audio_path),
                }
                if isinstance(segment_ids, dict):
                    events = segment_ids[segment_id]
                    if len(events) > 0 and isinstance(events[0], (list, tuple)):
                        events, event_onsets, event_offsets = list(zip(*events))
                        examples[segment_id][f'events_start_times'] = event_onsets
                        examples[segment_id][f'events_stop_times'] = event_offsets
                    elif purpose == "validation" and len(events) == 0:
                        examples[segment_id][f'events_start_times'] = []
                        examples[segment_id][f'events_stop_times'] = []
                    examples[segment_id]['events'] = events
            database['datasets'][f'desed_real_{name}'] = prepare_sound_dataset(examples)

            print(
                f'{len(segment_ids) - len(database["datasets"][f"desed_real_{name}"])} '
                f'from {len(segment_ids)} files missing in {name}.'
            )

    # desed_synthetic
    synthetic_dataset_path = database_path/'synthetic'
    audio_dir = synthetic_dataset_path/'audio'/'train'/'synthetic20'/'soundscapes'
    segment_ids = read_segments_file(synthetic_dataset_path/'audio'/'train'/'synthetic20'/'soundscapes.tsv')
    examples = {}
    for segment_id in natsorted(segment_ids):
        audio_path = audio_dir/f'{segment_id}.wav'
        examples[segment_id] = {
            'audio_path': str(audio_path),
        }
        assert isinstance(segment_ids, dict), type(segment_ids)
        events = segment_ids[segment_id]
        if len(events) > 0 and isinstance(events[0], (list, tuple)):
            events, event_onsets, event_offsets = list(zip(*events))
            examples[segment_id][f'events_start_times'] = event_onsets
            examples[segment_id][f'events_stop_times'] = event_offsets
        elif len(events) == 0:
            examples[segment_id][f'events_start_times'] = []
            examples[segment_id][f'events_stop_times'] = []
        examples[segment_id]['events'] = events
    database['datasets'][f'desed_synthetic'] = prepare_sound_dataset(examples)
    print(
        f'{len(segment_ids) - len(database["datasets"]["desed_synthetic"])} '
        f'from {len(segment_ids)} files missing in synthetic'
    )

    events = {
        event
        for ds in database['datasets'].values()
        for example in ds.values()
        for event in example.get('events', [])
    }
    print('Number of event labels:', len(events))

    # fuss
    # fuss_fsd_data_dir = database_path / 'fsd_data'
    # for txt_file in fuss_fsd_data_dir.glob('*.txt'):
    #     ds_name = txt_file.name[:-len('.txt')]
    #     examples = {}
    #     with txt_file.open() as fid:
    #         for audio_file in fid.read().splitlines():
    #             audio_file = fuss_fsd_data_dir / Path(audio_file)
    #             audio_id = audio_file.name[:-len('.wav')]
    #             examples[audio_id] = {
    #                 'audio_path': str(audio_file)
    #             }
    #     database['datasets'][f'fuss_{ds_name}'] = prepare_sound_dataset(
    #         examples
    #     )
    #
    # for ss_dir in ['ssdata', 'ssdata_reverb']:
    #     ss_dir = database_path / ss_dir
    #     for ds in ['train', 'validation', 'eval']:
    #         examples = {}
    #         for audio_file in (ss_dir / ds).glob('*.wav'):
    #             example_id = audio_file.name[:-len('.wav')]
    #
    #             examples[example_id] = {
    #                 'audio_path': str(audio_file),
    #             }
    #         examples = prepare_sound_dataset(examples)
    #         for example_id, example in examples.items():
    #             examples[example_id]['audio_path'] = {
    #                 'observation': example['audio_path'],
    #                 'background': str(ss_dir / ds / f'{example_id}_sources' / 'background0_sound.wav'),
    #                 'sources': sorted([str(src) for src in (ss_dir / ds / f'{example_id}_sources').glob('foreground*_sound.wav')]),
    #             }
    #         database['datasets'][f'fuss_{ss_dir.name}_{ds}'] = examples

    rir_dir = database_path / 'rir_data'
    for ds in rir_dir.iterdir():
        rooms = {}
        for room in (rir_dir / ds).iterdir():
            room_id = room.name.split('_')[1]
            rooms[room_id] = {
                'rirs': sorted([str(rir) for rir in room.glob('*.wav')])
            }
        database['datasets'][f'rir_data_{ds.name}'] = rooms
    return database


def read_segments_file(segment_file):
    examples = None
    with segment_file.open() as fid:
        for row in fid.read().splitlines()[1:]:
            row = row.split('\t')
            example_id = row[0][:-len('.wav')]
            if len(row) == 1:
                if examples is None:
                    examples = []
                else:
                    assert isinstance(examples, list)
                examples.append(example_id)
            elif len(row) == 2:
                if examples is None:
                    examples = defaultdict(list)
                else:
                    assert isinstance(examples, dict)
                examples[example_id].extend(row[1].split(','))
            elif len(row) == 4:
                if examples is None:
                    examples = defaultdict(list)
                else:
                    assert isinstance(examples, defaultdict)
                if len(row[3]) > 0:
                    examples[example_id].append(
                        [row[3], float(row[1]), float(row[2])]
                    )
                else:
                    examples[example_id] = []
            else:
                raise Exception
    return examples


def create_json(database_path: Path, json_path: Path, indent=4):
    assert database_path.is_dir(), (
        f'Path "{str(database_path.absolute())}" is not a directory.'
    )
    database = construct_json(database_path)
    dump_json(
        database,
        json_path,
        create_path=True,
        indent=indent,
        ensure_ascii=False,
    )


@click.command()
@click.option(
    '--database-path', '-db',
    help='Path where the database is located.',
    type=click.Path(),
)
@click.option(
    '--json-path', '-j',
    default=str(database_jsons_dir / 'desed.json'),
    help=f'Output path for the generated JSON file. If the file exists, it '
         f'gets overwritten. Defaults to '
         f'"{database_jsons_dir / "desed.json"}".',
    type=click.Path(dir_okay=False, writable=True),
)
def main(database_path, json_path):
    create_json(Path(database_path), Path(json_path))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
