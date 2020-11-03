"""
This script downloads the DESED database to a local path with the following structure:

├── real
│   ├── audio
│   │   ├── eval
│   │   │   ├── eval_dcase2019
│   │   │   └── eval_dcase2020
│   │   ├── train
│   │   │   ├── unlabel_in_domain
│   │   │   └── weak
│   │   └── validation
│   │       └── validation
│   ├── dataset
│   │   ├── audio
│   │   │   └── eval
│   │   └── metadata
│   │       └── eval
│   ├── metadata
│   │   ├── eval
│   │   ├── train
│   │   └── validation
│   └── missing_files
├── rir_data
│   ├── eval
│   ├── train
│   └── validation
└── synthetic
    ├── audio
    │   ├── eval
    │   │   └── soundbank
    │   └── train
    │       ├── soundbank
    │       └── synthetic20
    ├── dcase2019
    │   └── dataset
    │       ├── audio
    │       └── metadata
    └── metadata
        └── train
            └── synthetic20

Information about the database can be found here:
https://project.inria.fr/desed/dcase-challenge/dcase-2020-task-4/

Example usage:
python -m pb_sed.database.desed.download -db /desired/path/to/desed

"""
import os
from pathlib import Path
from shutil import copyfile
from urllib.request import urlretrieve

import click
import pandas as pd
from desed.download_real import download
from desed.generate_synthetic import generate_files_from_jams, \
    generate_tsv_from_jams
from desed.get_backgroung_training import get_background_training
from paderbox.io.download import download_file_list
from pb_sed.paths import pb_sed_root


def _download_real_metadata(dataset_path):
    dataset_path = Path(dataset_path)
    os.makedirs(str(dataset_path / 'metadata' / 'train'), exist_ok=True)
    os.makedirs(str(dataset_path / 'metadata' / 'validation'), exist_ok=True)
    remote = 'https://raw.githubusercontent.com/turpaultn/DESED/master/real/metadata'
    for filename in ['unlabel_in_domain.tsv', 'weak.tsv']:
        remote_file = f"{remote}/train/{filename}"
        local_file = str(dataset_path / 'metadata' / 'train' / filename)
        print("Download", remote_file)
        urlretrieve(remote_file, local_file)
    remote_file = f"{remote}/validation/validation.tsv"
    local_file = str(dataset_path / 'metadata' / 'validation' / 'validation.tsv')
    print("Download", remote_file)
    urlretrieve(remote_file, local_file)


def download_real_audio_from_csv(
        csv_path, audio_dir, n_jobs, chunk_size, base_missing_files_folder
):
    # read metadata file and get only one filename once
    df = pd.read_csv(csv_path, header=0, sep='\t')
    filenames_test = df["filename"].drop_duplicates()
    download(
        filenames_test,
        audio_dir,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        base_dir_missing_files=base_missing_files_folder
    )


@click.command()
@click.option(
    '--database_path',
    '-db',
    type=str,
    default=str(Path('.') / 'DESED'),
    help=f'Destination directory for the database. Defaults to '
    f'"{Path(".") / "DESED"}"'
)
@click.option(
    '--n_jobs',
    '-j',
    type=int,
    default=3,
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

    # DESED
    # Download real
    dataset_real_path = database_path / 'real'
    _download_real_metadata(dataset_real_path)
    download_real_audio_from_csv(
        str(dataset_real_path / "metadata" / "validation" / "validation.tsv"),
        str(dataset_real_path / "audio" / "validation" / "validation"),
        n_jobs,
        chunk_size,
        str(dataset_real_path)
    )
    download_real_audio_from_csv(
        str(dataset_real_path / "metadata" / "train" / "weak.tsv"),
        str(dataset_real_path / "audio" / "train" / "weak"),
        n_jobs,
        chunk_size,
        str(dataset_real_path)
    )
    download_real_audio_from_csv(
        str(dataset_real_path / "metadata" / "train" / "unlabel_in_domain.tsv"),
        str(dataset_real_path / "audio" / "train" / "unlabel_in_domain"),
        n_jobs,
        chunk_size,
        str(dataset_real_path)
    )
    download_file_list(
        [
            "https://zenodo.org/record/3866455/files/eval.tar.gz",
            "https://zenodo.org/record/3588172/files/DESEDpublic_eval.tar.gz"
        ],
        dataset_real_path
    )
    os.mkdir(str(dataset_real_path / "metadata" / "eval"))
    (dataset_real_path / "metadata" / "eval.tsv").rename(
        dataset_real_path / "metadata" / "eval" / "eval_dcase2020.tsv"
    )
    (dataset_real_path / "dataset" / "metadata" / "eval" / "public.tsv").rename(
        dataset_real_path / "metadata" / "eval" / "eval_dcase2019.tsv"
    )
    for timestamp in [
        '2020-07-03-20-48-45', '2020-07-03-20-49-48', '2020-07-03-20-52-19',
        '2020-07-03-21-00-48', '2020-07-03-21-05-34',
        '2020-07-04-13-10-05', '2020-07-04-13-10-19', '2020-07-04-13-10-33',
        '2020-07-04-13-11-09', '2020-07-04-13-12-06',
        '2020-07-05-12-37-18', '2020-07-05-12-37-26', '2020-07-05-12-37-35',
        '2020-07-05-12-37-45', '2020-07-05-12-37-54',
    ]:
        for file in (pb_sed_root / 'exp' / 'dcase_2020_inference' / timestamp).glob('*.tsv'):
            if file.name.startswith('weak') or file.name.startswith('unlabel_in_domain'):
                copyfile(
                    file, dataset_real_path / "metadata" / "train" / file.name,
                )

    (dataset_real_path / "audio" / "eval").rename(
        dataset_real_path / "audio" / "eval_dcase2020"
    )
    os.mkdir(str(dataset_real_path / "audio" / "eval"))
    (dataset_real_path / "audio" / "eval_dcase2020").rename(
        dataset_real_path / "audio" / "eval" / "eval_dcase2020"
    )
    (dataset_real_path / "dataset" / "audio" / "eval" / "public").rename(
        dataset_real_path / "audio" / "eval" / "eval_dcase2019"
    )

    # Download Synthetic
    dataset_synthetic_path = database_path / 'synthetic'
    # Soundbank
    download_file_list(
        [
            "https://zenodo.org/record/3713328/files/DESED_synth_soundbank.tar.gz",
        ],
        database_path
    )
    # Backgrounds
    get_background_training(dataset_synthetic_path, sins=True, tut=False, keep_sins=False)

    download_file_list(
        [
            # "https://zenodo.org/record/3713328/files/DESED_synth_soundbank.tar.gz",
            "https://zenodo.org/record/3713328/files/DESED_synth_dcase20_train_jams.tar.gz",
            "https://zenodo.org/record/3713328/files/DESED_synth_eval_dcase2019.tar.gz"
        ],
        dataset_synthetic_path
    )
    out_dir = dataset_synthetic_path / 'audio' / 'train' / 'synthetic20' / 'soundscapes'
    list_jams = [str(p) for p in out_dir.glob("*.jams")]
    fg_path_train = str(dataset_synthetic_path / 'audio' / 'train' / 'soundbank' / 'foreground')
    bg_path_train = str(dataset_synthetic_path / 'audio' / 'train' / 'soundbank' / 'background')
    out_tsv = str(dataset_synthetic_path / 'audio' / 'train' / 'synthetic20' / 'soundscapes.tsv')
    generate_files_from_jams(
        list_jams, str(out_dir), out_folder_jams=str(out_dir),
        fg_path=fg_path_train, bg_path=bg_path_train,
        save_isolated_events=True
    )
    generate_tsv_from_jams(list_jams, out_tsv)

    # FUSS
    download_file_list(
        [
            "https://zenodo.org/record/3694384/files/FUSS_rir_data.tar.gz",
            # "https://zenodo.org/record/3694384/files/FUSS_fsd_data.tar.gz",
            # "https://zenodo.org/record/3694384/files/FUSS_ssdata.tar.gz",
            # "https://zenodo.org/record/3694384/files/FUSS_ssdata_reverb.tar.gz"
        ],
        database_path
    )


if __name__ == '__main__':
    main()
