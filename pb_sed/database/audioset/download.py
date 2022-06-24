import csv
import json
import os
import subprocess
import sys
import time
import timeout_decorator
from contextlib import contextmanager
from multiprocessing import Queue, Process
from pathlib import Path
from shutil import copyfile
from urllib.request import urlretrieve
from tqdm import tqdm

import click
import yt_dlp


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def _format_audio(
        input_file, output_file, start_time, end_time, remove_input_file=True):
    assert input_file != output_file
    cmdstring = f"ffmpeg -loglevel error -i {input_file} -ac 1 -ar 44100 " \
                f"-ss {start_time} -to {end_time} -y {output_file}"
    subprocess.check_output(
        cmdstring, shell=True, stderr=subprocess.DEVNULL, timeout=60.,
    )
    if remove_input_file:
        Path(input_file).unlink()


@timeout_decorator.timeout(120.)
def _download_clip(clip_id, start, end, ydl, audio_folder, info_folder):
    url = "https://www.youtube.com/watch?v=" + clip_id
    output_file = audio_folder / (clip_id + ".wav")
    audio_available = output_file.exists()
    info_file = info_folder / (clip_id + ".json")
    info_available = info_file.exists()
    if audio_available and info_available:
        return

    info = ydl.extract_info(url, download=not audio_available)
    with info_file.open('w') as fid:
        json.dump(info, fid, indent=4, sort_keys=True)
    if not audio_available:
        download_file = audio_folder / (clip_id + "_temp." + info['ext'])
        _format_audio(str(download_file), str(output_file), start, end)


def _worker(
        input_queue: Queue, output_queue: Queue, output_folder, info_folder, *,
        cookiefile=None, verbose=False, timeout=5,
):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_folder / '%(id)s_temp.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': timeout,
    }
    if cookiefile is not None:
        ydl_opts['cookiefile'] = cookiefile
    with suppress_stderr(), yt_dlp.YoutubeDL(ydl_opts) as ydl:
        while not input_queue.empty():
            clips = input_queue.get()
            for clip_id, start, end in clips:
                try:
                    _download_clip(clip_id, start, end, ydl, output_folder, info_folder)
                    output_queue.put((clip_id, True))
                except Exception as ex:
                    if verbose:
                        print(str(clip_id) + " --> " + str(type(ex)) + ": " + str(ex))
                    output_queue.put((clip_id, False))


def download_clips(
        clips, output_folder, info_folder, num_jobs=8, chunksize=20,
        cookiefile=None, verbose=False
):
    os.makedirs(str(output_folder), exist_ok=True)
    os.makedirs(str(info_folder), exist_ok=True)
    # available_ids = {
    #     wav_file.name.split('.wav')[0]
    #     for wav_file in output_folder.glob('*.wav')
    # }
    # clips = [clip for clip in clips if clip[0] not in available_ids]
    chunks = [
        clips[i:i+chunksize]
        for i in range(0, len(clips), chunksize)
    ]

    input_queue = Queue()
    for chunk in chunks:
        input_queue.put(chunk)
    output_queue = Queue()
    workers = []
    for i in range(num_jobs):
        worker_kwargs = {'verbose': verbose}
        if cookiefile is not None:
            copyfile(Path(cookiefile).expanduser(), f"/tmp/cookies{i}.txt")
            worker_kwargs['cookiefile'] = f"/tmp/cookies{i}.txt"
        workers.append(
            Process(
                target=_worker,
                args=(input_queue, output_queue, output_folder, info_folder),
                kwargs=worker_kwargs,
                daemon=True
            )
        )
        workers[i].start()

    pbar = tqdm(initial=0, total=len(clips))
    try:
        while not input_queue.empty():
            while not output_queue.empty():
                output_queue.get()
                pbar.update(1)
            time.sleep(1)
        time.sleep(20)
        while not output_queue.empty():
            output_queue.get()
            pbar.update(1)
    finally:
        pbar.close()


def download_clips_from_csv(
        csv_file, output_folder, info_folder, num_jobs=8, chunksize=20,
        cookiefile=None, verbose=False
):
    with csv_file.open() as fid:
        clips = [
            (row[0], row[1], row[2]) for row in csv.reader(fid)
            if len(row) > 0 and not row[0].startswith('#')
        ]
    download_clips(
        clips, output_folder, info_folder, num_jobs=num_jobs, chunksize=chunksize,
        cookiefile=cookiefile, verbose=verbose
    )


@click.command()
@click.option(
    '--datasets',
    '-ds',
    type=str,
    default='balanced_train,eval,unbalanced_train',
    help='String of datasets separated by ","'
)
@click.option(
    '--database_path',
    '-db',
    type=str,
    default='AudioSet',
    help='Destination directory for the database'
)
@click.option(
    '--num_jobs',
    '-j',
    type=int,
    default=8,
    help='Number of parallel download jobs'
)
@click.option(
    '--cookiefile',
    '-c',
    type=str,
    default=None,
    help='Optional path to cookiefile for youtube-dl'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help="Print more output."
)
def download(
        datasets,
        database_path,
        num_jobs,
        cookiefile,
        verbose,
):
    datasets = datasets.split(',')
    os.makedirs(database_path, exist_ok=True)
    database_path = Path(database_path).expanduser().absolute()

    remote = 'http://storage.googleapis.com/us_audioset/youtube_corpus'
    remote_files = [
        remote + "/v1/csv/" + (dataset + '_segments.csv') for dataset in datasets
    ]
    remote_files.append(remote + "/v1/csv/" + 'class_labels_indices.csv')
    remote_files.append(remote + '/strong/audioset_train_strong.tsv')
    remote_files.append(remote + '/strong/audioset_eval_strong.tsv')
    remote_files.append(remote + '/strong/mid_to_display_name.tsv')
    local_files = [database_path / file.split("/")[-1] for file in remote_files]
    for remote_file, local_file in zip(remote_files, local_files):
        urlretrieve(remote_file, str(local_file))
    audio_dir = database_path / 'audio'
    info_dir = database_path / 'info'
    for dataset_name, csv_file in zip(datasets, local_files[:len(datasets)]):
        print(f'Download {dataset_name}.')
        download_clips_from_csv(
            csv_file=csv_file,
            output_folder=audio_dir / dataset_name,
            info_folder=info_dir / dataset_name,
            num_jobs=num_jobs,
            cookiefile=cookiefile,
            verbose=verbose,
        )


if __name__ == "__main__":
    download()
