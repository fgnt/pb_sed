# resample audio set files
import shutil
import click
import paderbox as pb
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import soundfile as sf


def process_file(
        source_path, destination_path, relative_path,
        audio_output_rate, audio_output_format,
        skip_existing=False, excluded=(),
):
    source_file_path = source_path / relative_path
    if source_file_path.is_dir():
        return False
    if any([relative_path.startswith(excl) for excl in excluded]):
        return False
    source_file_split = source_file_path.name.rsplit(
        '.', maxsplit=1)
    if len(source_file_split) == 1:
        source_file_stem, = source_file_split
        source_file_suffix = ''
    else:
        source_file_stem, source_file_suffix = source_file_split
    if not source_file_stem:
        return False
    destination_file_path = destination_path / relative_path
    if not destination_file_path.parent.exists():
        destination_file_path.parent.mkdir(parents=True, exist_ok=True)
    if source_file_suffix == 'wav':
        destination_file_path = (
            destination_file_path.parent
            / f"{source_file_stem}.{audio_output_format}"
        )
        if destination_file_path.exists() and skip_existing:
            return True
        try:
            with sf.SoundFile(str(source_file_path)) as f:
                detected_in_rate = f.samplerate
        except sf.LibsndfileError:
            return False
        if audio_output_rate == detected_in_rate and audio_output_format == source_file_suffix:
            shutil.copy(str(source_file_path), str(destination_file_path))
        else:
            audio_data = pb.io.load(source_file_path)
            if (audio_data.size == 0) or (np.max(np.abs(audio_data)) < 1e-12):
                return False
            if audio_output_rate != detected_in_rate:
                audio_data = pb.transform.resample_sox(
                    audio_data, in_rate=detected_in_rate, out_rate=audio_output_rate
                )
            pb.io.dump_audio(audio_data, destination_file_path, sample_rate=audio_output_rate, format=audio_output_format)
    elif not destination_file_path.exists() or not skip_existing:
        shutil.copy(str(source_file_path), str(destination_file_path))
    return True


def print_example(
        source_path, destination_path, relative_path,
        audio_output_rate, audio_output_format,
        skip_existing=False, excluded=(),
):
    source_file_path = source_path / relative_path
    if source_file_path.is_dir():
        return False
    if any([relative_path.startswith(excl) for excl in excluded]):
        return False
    source_file_split = source_file_path.name.rsplit(
        '.', maxsplit=1)
    if len(source_file_split) == 1:
        source_file_stem, = source_file_split
        source_file_suffix = ''
    else:
        source_file_stem, source_file_suffix = source_file_split
    destination_file_path = destination_path / relative_path
    if source_file_suffix in ['wav', 'flac']:
        destination_file_path = (
            destination_file_path.parent
            / f"{source_file_stem}.{audio_output_format}"
        )
        if destination_file_path.exists() and skip_existing:
            print(f"{destination_file_path} already exists")
        else:
            print(f"write (resampled) {source_file_path} to {destination_file_path}")
    elif destination_file_path.exists() and skip_existing:
        print(f"{destination_file_path} already exists")
    else:
        print(f"copy {source_file_path} to {destination_file_path}")
    return True


@click.command()
@click.option("--source_path", "-s", help="Path to original database directory")
@click.option("--destination_path", "-d", help="Path of the new database root directory")
@click.option("--out_rate", "-r", default=16000, help="Desired sample rate")
@click.option("--out_format", "-f", default='flac', help="Desired output format")
@click.option("--excluded", "-e", default=None, help="Excluded sub paths separated by comma.")
@click.option("--num_workers", "-w", default=50, help="Number of resampling threads")
@click.option(
    "--dry",
    is_flag=True,
    show_default=True,
    default=False,
    help="Dry run, shows which files are resampled and where they are saved",
)
@click.option(
    "-b",
    "--batch_size",
    default=100,
    help="If positive integer, size of the batches, otherwise no batching if None",
)
@click.option("--exist_ok",
              is_flag=True,
              default=False,
              help="Continue if destination directory exists")
@click.option("--skip_existing", 'skip_existing', is_flag=True, default=True,
              help="Continue resampling, skipping all existing files")
def resample_db(
    source_path,
    destination_path,
    out_rate,
    out_format,
    excluded,
    num_workers,
    dry,
    batch_size,
    exist_ok,
    skip_existing,
):
    source_path = Path(source_path).expanduser().absolute()
    destination_path = Path(destination_path).expanduser().absolute()

    assert source_path.exists(), f"Source database not found at {source_path}"
    assert source_path.is_dir(), f"The given source directory {source_path} is not a directory."
    assert exist_ok or not destination_path.exists(), f"Destination already exists: {destination_path}"

    if dry:
        print("dry run")
    else:
        destination_path.mkdir(exist_ok=exist_ok, parents=True)
    file_list = list(source_path.rglob('*'))

    print(f'Found {len(file_list)} files in source directory')

    excluded = [] if excluded is None else excluded.split(',')
    relative_file_paths = [str(file_path.relative_to(source_path)) for file_path in file_list]

    if batch_size is None:
        batch_size = 1
    else:
        batch_size = int(batch_size)

    # load/resample/write files
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_fn = process_file
        if dry:
            relative_file_paths = relative_file_paths[:100]
            process_fn = print_example
        num_tasks = len(relative_file_paths)
        n_files = sum(list(
            tqdm(
                executor.map(
                    process_fn,
                    num_tasks * [source_path],
                    num_tasks * [destination_path],
                    relative_file_paths,
                    num_tasks * [out_rate],
                    num_tasks * [out_format],
                    num_tasks * [skip_existing],
                    num_tasks * [excluded],
                    chunksize=batch_size,
                ),
                total=num_tasks,
            )
        ))
    print(f'Resampled database created at {destination_path} with {n_files} files')


if __name__ == "__main__":
    resample_db()
