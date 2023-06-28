# resample audio set files
import shutil
import click
import paderbox as pb
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import soundfile as sf


def process_example(source_path, destination_path, relative_audio_path, out_rate):
    source_audio_path = source_path/relative_audio_path
    if not source_audio_path.suffix == ".wav" or not source_audio_path.exists():
        print(f"skipping {source_audio_path} because it is not a wav file")
        return None
    destination_audio_path = destination_path/relative_audio_path
    with sf.SoundFile(str(source_audio_path)) as f:
        detected_in_rate = f.samplerate

    if out_rate == detected_in_rate:
        if not destination_audio_path.parent.exists():
            destination_audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source_audio_path), str(destination_audio_path))
    else:
        audio_data = pb.io.load(source_audio_path)
        resampled_audio_data = pb.transform.resample_sox(
            audio_data, in_rate=detected_in_rate, out_rate=out_rate
        )
        if not destination_audio_path.exists():
            pb.io.dump(resampled_audio_data, destination_audio_path, mkdir=True, mkdir_exist_ok=True, mkdir_parents=True)


def print_example(source_path, destination_path, relative_audio_path):
    source_audio_path = source_path/relative_audio_path
    destination_audio_path = destination_path/relative_audio_path
    print(f"write resampled {source_audio_path} to {destination_audio_path}")
    

@click.command()
@click.option("--source_path", help="Path to original database directory")
@click.option("--destination_path", help="Path of the new database root directory")
@click.option("--out_rate", default=16000, help="Desired sample rate")
@click.option("--num_workers", default=50, help="Number of resampling threads")
@click.option(
    "-d",
    "--dry",
    is_flag=True,
    show_default=True,
    default=False,
    help="Dry run, shows which files are resampled and where they are saved",
)
@click.option(
    "--batching",
    default=100,
    help="If positive integer, size of the batches, otherwise no batching if None",
)
@click.option("--exist_ok",
              is_flag=True,
              default=False,
              help="Continue if destination directory exists")
@click.option("--audio_file_glob",
              default='audio/**/*.wav',
              help="Glob used to find audio files inside the database directory")
@click.option("--continue", 'continue_', is_flag=True, default=False,
             help="Continue resampling, skipping all existing files")
def resample_db(
    source_path,
    destination_path,
    out_rate,
    num_workers,
    dry,
    batching,
    exist_ok,
    audio_file_glob,
    continue_,
):
    source_path = Path(source_path)
    assert source_path.exists(), f"Source database not found at {source_path}"
    if not exist_ok and not continue_:
        assert not Path(destination_path).exists(), f"Destination already exists: {destination_path}"

    if source_path.is_dir():
        source_db_path = source_path
        audio_path_list = source_db_path.rglob(audio_file_glob)
    else:
        raise ValueError(f"The given source directory {source_path} is not a directory.")

    # ignore existing files, this does not check for correctess of the file content
    if continue_:
        existing_files = set([p.relative_to(destination_path) for p in Path(destination_path).rglob(audio_file_glob)])

    relative_audio_paths = []
    for audio_path in audio_path_list:
        relative_path = audio_path.relative_to(source_path)
        if not continue_ or relative_path not in existing_files:
           relative_audio_paths.append(relative_path)

    if batching is not None:
        try:
            batching = int(batching)
            if batching > 1:
                batch_size = batching
        except ValueError:
            pass
    else:
        batch_size = 1

    if dry:
        print("dry run")
    else:
        Path(destination_path).mkdir(exist_ok=True, parents=True)

    # load/resample/write files
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_fn = process_example
        if dry:
            relative_audio_paths = relative_audio_paths[:10]
            process_fn = print_example
        num_tasks = len(relative_audio_paths)
        list(
            tqdm(
                executor.map(
                    process_fn,
                    num_tasks * [source_path],
                    num_tasks * [destination_path],
                    relative_audio_paths,
                    num_tasks * [out_rate],
                    chunksize=batch_size,
                ),
                total=num_tasks,
            )
        )
    # copy annotation files
    annotation_files = source_db_path.glob("*")
    if dry:
        annotation_files = []
    for annotation_file in annotation_files:
        if annotation_file.is_file():
            file_name = annotation_file.name
            shutil.copy(str(annotation_file), str(destination_path/file_name))

    print(f'Resampled database created at {destination_path}')


if __name__ == "__main__":
    resample_db()
