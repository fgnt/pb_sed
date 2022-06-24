"""
This script downloads the DESED database to a local path with the following structure:
├── audio
│   ├── eval
│   │   └── public
│   ├── train
│   │   ├── unlabel_in_domain
│   │   └── weak
│   └── validation
│       └── validation
├── metadata
│   ├── eval
│   ├── train
│   └── validation
└── missing_files

Information about the database can be found here:
https://project.inria.fr/desed/dcase-challenge/dcase-2020-task-4/

Example usage:
python -m pb_sed.database.desed.download -db /desired/path/to/desed

"""

import click
import desed
from desed.download import split_desed_soundbank_train_val
from pathlib import Path
import shutil
from paderbox.io.download import download_file_list


@click.command()
@click.option(
    '--database_path',
    '-db',
    type=str,
    default='./DESED',
    help=f'Base directory for the databases. Defaults to "./DESED"'
)
@click.option(
    '--n_jobs',
    '-j',
    type=int,
    default=8,
)
@click.option(
    '--chunk_size',
    '-c',
    type=int,
    default=10,
)
def main(database_path, n_jobs, chunk_size):
    """Download dataset packages over the internet to a local path

    Args:
        database_path:
        n_jobs:
        chunk_size:

    Returns:

    """
    database_path = Path(database_path).absolute()
    database_path.mkdir(parents=True, exist_ok=True)

    # ##########
    # Real data
    # ##########
    desed.download.download_real(
        str(database_path),
        n_jobs=n_jobs, chunk_size=chunk_size,
        eval=not (database_path / 'audio' / 'eval' / 'public').exists(),
    )
    (database_path / 'metadata' / 'validation' / 'test_dcase2018.tsv').unlink()
    (database_path / 'metadata' / 'validation' / 'eval_dcase2018.tsv').unlink()
    (database_path / 'metadata' / 'validation' / '._test_dcase2018.tsv').unlink()
    (database_path / 'metadata' / 'validation' / '._eval_dcase2018.tsv').unlink()
    shutil.move(str(Path('missing_files').absolute()), str(database_path / 'missing_files'))
    download_file_list(
        ["https://zenodo.org/record/6444477/files/audioset_strong.tsv"],
        database_path / 'metadata' / 'train'
    )
    (database_path / 'metadata' / 'train' / 'audioset_strong.tsv').absolute().rename(
        database_path / 'metadata' / 'train' / 'strong.tsv')
    desed.download.download_audioset_files_from_csv(
        str(database_path / 'metadata' / 'train' / 'strong.tsv'),
        str(database_path / "audio" / "train" / "strong"),
        missing_files_tsv=str(database_path / 'missing_files' / "missing_files_strong.tsv"),
        n_jobs=n_jobs,
    )

    # ##########
    # Synthetic soundscapes DCASE 2020
    # ##########
    synthetic_path = database_path / 'synthetic'
    soundbank20_path = synthetic_path / 'soundbank20'
    jams20_path = synthetic_path / 'jams20'
    # Generate audio files. We can loop because it is the same structure of folders for the three sets.
    for purpose in ["train", "validation", "eval"]:
        # Download the soundbank if needed
        if not soundbank20_path.exists():
            desed.download.download_desed_soundbank(
                str(soundbank20_path), sins_bg=True, tut_bg=True
            )
        elif not (soundbank20_path / "audio" / "validation").exists():
            # If you don't have the validation split, rearrange the soundbank in train-valid (split in 90%/10%)
            split_desed_soundbank_train_val(str(soundbank20_path))
        # Download jams if needed
        if not jams20_path.exists():
            download_file_list(
                [
                    "https://zenodo.org/record/6026841/files/DESED_synth_dcase20_train_val_jams.tar.gz",
                    "https://zenodo.org/record/6026841/files/DESED_synth_dcase20_eval_jams.tar.gz",
                ],
                jams20_path
            )

        audio_source_path = jams20_path / "audio" / purpose / ("synthetic20_" + purpose) / "soundscapes"
        list_jams = [str(f) for f in audio_source_path.glob("*.jams")]
        fg_path = soundbank20_path / "audio" / purpose / "soundbank" / "foreground"
        bg_path = soundbank20_path / "audio" / purpose / "soundbank" / "background"
        out_tsv = database_path / "metadata" / purpose / "synthetic20.tsv"
        target_path = database_path / 'audio' / purpose / 'synthetic20'

        desed.generate_files_from_jams(
            list_jams,
            fg_path=fg_path,
            bg_path=bg_path,
            out_folder=target_path,
            out_folder_jams=None,
            save_isolated_events=False,
            overwrite_exist_audio=False,
        )
        desed.generate_tsv_from_jams(list_jams, str(out_tsv))

    # ##########
    # Synthetic soundscapes DCASE 2021
    # ##########
    synthetic21_path = synthetic_path / 'dcase_synth'
    for purpose in ["train", "validation"]:
        audio_target_path = database_path / 'audio' / purpose / 'synthetic21'
        if audio_target_path.exists():
            continue
        if not synthetic21_path.exists():
            download_file_list(
                ["https://zenodo.org/record/6026841/files/dcase_synth.zip"],
                synthetic_path
            )
        audio_source_path = synthetic21_path / "audio" / purpose / ("synthetic21_" + purpose) / "soundscapes"
        for file in audio_source_path.glob("*.jams"):
            file.unlink()
        for file in audio_source_path.glob("*.txt"):
            file.unlink()
        audio_source_path.rename(audio_target_path)
        ground_truth_file = synthetic21_path / 'metadata' / purpose / ("synthetic21_" + purpose) / "soundscapes.tsv"
        ground_truth_file.rename(database_path / "metadata" / purpose / "synthetic21.tsv")


if __name__ == '__main__':
    main()
