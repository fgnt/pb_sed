import csv
import re
import click
from copy import deepcopy
from pathlib import Path

from paderbox.io.json_module import dump_json
from pb_sed.database.helper import prepare_sound_dataset
from pb_sed.paths import database_jsons_dir


def construct_json(database_path: Path) -> dict:
    datasets = {}
    weak_event_classes = set()
    strong_event_classes = set()

    # load mapping for strong labels
    with (database_path / 'mid_to_display_name.tsv').open() as f:
        mid_to_display_name = {
            row[0]: row[1] for row in csv.reader(f, delimiter='\t')
        }

    train_strong_labels = read_strong_label_files(
        database_path / 'audioset_train_strong.tsv',
        mapping=mid_to_display_name,
    )
    train_strong_examples = {}
    eval_strong_labels = read_strong_label_files(
        database_path / 'audioset_eval_strong.tsv',
        mapping=mid_to_display_name,
    )
    eval_strong_examples = {}

    # load mapping for weak labels
    with (database_path / 'class_labels_indices.csv').open() as f:
        mid_to_display_name = {
            row['mid']: row['display_name'].strip('"')
            for row in csv.DictReader(f)
        }
    # iterate over all segment files (datasets)
    for segment_file in sorted(database_path.glob('*_segments.csv')):
        # read segment file
        name = segment_file.name.replace('_segments.csv', '')
        tags_dict = read_weak_label_file(
            segment_file, mid_to_display_name
        )
        audio_dir = database_path / 'audio' / name
        examples = {}
        for clip_id, tags in tags_dict.items():
            examples[clip_id] = {
                'audio_path': audio_dir / f'{clip_id}.wav',
                'events': tags,
            }
            weak_event_classes.update(tags)
        datasets[name], missing = prepare_sound_dataset(examples)

        if 'eval' in name:
            strong_labels = eval_strong_labels
            strong_examples = eval_strong_examples
        else:
            strong_labels = train_strong_labels
            strong_examples = train_strong_examples
        n_strong_labels = 0
        n_matching_strong_labels = 0
        for clip_id in datasets[name]:
            if clip_id in strong_labels:
                strong_examples[clip_id] = deepcopy(datasets[name][clip_id])
                onsets, offsets, events = list(zip(*strong_labels[clip_id]))
                n_strong_labels += len(events)
                strong_event_classes.update(events)
                strong_examples[clip_id].update({
                    'events': list(events),
                    'events_start_times': list(onsets),
                    'events_stop_times': list(offsets),
                })
                tags = datasets[name][clip_id]["events"]
                matching_strong_labels = [
                    event for event in strong_labels[clip_id]
                    if event[-1] in tags
                ]
                n_matching_strong_labels += len(matching_strong_labels)
                additional_weak_labels = [
                    (0., datasets[name][clip_id].get('audio_length', 10.), tag)
                    for tag in tags if tag not in events
                ]
                onsets, offsets, events = list(zip(*(additional_weak_labels + matching_strong_labels)))
                label_types = len(additional_weak_labels) * ["weak"] + len(matching_strong_labels) * ["strong"]
                datasets[name][clip_id].update({
                    'events': list(events),
                    'events_start_times': list(onsets),
                    'events_stop_times': list(offsets),
                    'label_types': label_types,
                })
        print(f'{n_strong_labels} strong labels in {name}.')
        print(f'{n_matching_strong_labels} matching strong labels in {name}.')

        with Path(f'audioset_{name}_missing.txt').open('w') as fid_mis:
            with Path(f'audioset_{name}_damaged.txt').open('w') as fid_dmg:
                fid = (fid_mis, fid_dmg)
                n = [0, 0]
                for clip_id in sorted(missing):
                    if (audio_dir / f'{clip_id}.wav').exists():
                        idx = 1
                        line = str(audio_dir.absolute() / f'{clip_id}.wav')
                    else:
                        idx = 0
                        line = f'{clip_id}.wav'
                    if n[idx] > 0:
                        line = '\n' + line
                    fid[idx].write(line)
                    n[idx] += 1
        print(f'{n[0]} from {len(examples)} files missing in {name}')
        print(f'{n[1]} from {len(examples)} files damaged in {name}')

    datasets['train_strong'] = train_strong_examples
    datasets['eval_strong'] = eval_strong_examples
    print('Number of weak event classes:', len(weak_event_classes))
    print('Number of strong event classes:', len(strong_event_classes))
    return {'datasets': datasets}


def read_weak_label_file(csv_file, mapping):
    with csv_file.open() as fid:
        tags = {
            row[0]: [
                mapping[tag] for tag in re.findall(
                    r'/[mtg]/[\d_a-z]+', ''.join(row[1:]))
            ]
            for row in csv.reader(fid) if not row[0].startswith('#')
        }
    return tags


def read_strong_label_files(tsv_file, mapping):
    strong_labels = {}
    with tsv_file.open() as fid:
        for i, row in enumerate(csv.reader(fid, delimiter='\t')):
            if i == 0:
                continue
            clip_id, onset, offset, event_label = row
            clip_id = clip_id.rsplit('_', maxsplit=1)[0]
            if clip_id not in strong_labels:
                strong_labels[clip_id] = []
            strong_labels[clip_id].append(
                (float(onset), float(offset), mapping[event_label]))
    for clip_id in strong_labels:
        strong_labels[clip_id] = sorted(strong_labels[clip_id])
    return strong_labels


def create_json(database_path: Path, json_path: Path, indent=4):
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
    default=str(database_jsons_dir / 'audioset.json'),
    help=f'Output path for the generated JSON file. If the file exists, it '
         f'gets overwritten. Defaults to '
         f'"{database_jsons_dir / "audio_set.json"}".',
    type=click.Path(dir_okay=False, writable=True),
)
def main(database_path, json_path):
    create_json(
        Path(database_path).expanduser().absolute(), Path(json_path),
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
