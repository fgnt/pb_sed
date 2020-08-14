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
    """Download dataset packages over the internet to the local path

    Parameters
    ----------

    Returns
    -------
    Nothing

    Raises
    -------
    IOError
        Download failed.

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
    copyfile(
        pb_sed_root / 'exp' / 'dcase_2020_tagging' / '2020-07-03-22-27-00' / 'unlabel_in_domain_pseudo_weak_2020-07-03-22-27-00.tsv',
        dataset_real_path / "metadata" / "train" / 'unlabel_in_domain_pseudo_weak_2020-07-03-22-27-00.tsv',
    )
    copyfile(
        pb_sed_root / 'exp' / 'dcase_2020_detection' / '2020-07-04-22-16-46' / 'weak_pseudo_strong_2020-07-04-22-16-46.tsv',
        dataset_real_path / "metadata" / "train" / 'weak_pseudo_strong_2020-07-04-22-16-46.tsv',
    )
    copyfile(
        pb_sed_root / 'exp' / 'dcase_2020_detection' / '2020-07-04-22-33-13' / 'unlabel_in_domain_pseudo_strong_2020-07-04-22-33-13.tsv',
        dataset_real_path / "metadata" / "train" / 'unlabel_in_domain_pseudo_strong_2020-07-04-22-33-13.tsv',
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
    copyfile(
        pb_sed_root / 'exp' / 'dcase_2020_tagging' / '2020-07-03-22-27-00' / 'unlabel_in_domain_pseudo_weak_2020-07-03-22-27-00.tsv',

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
