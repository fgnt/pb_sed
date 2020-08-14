from pathlib import Path

import numpy as np
import torch
from paderbox.io.json_module import load_json
from paderbox.utils.timer import timeStamped
from padercontrib.evaluation.event_detection import fscore
from padertorch.data import example_to_device
from pb_sed.experiments.dcase_2020_task_4 import data
from pb_sed.models.fbcrnn import FBCRNN
from pb_sed.paths import storage_root
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver

ex_name = 'dcase_2020_tagging'
ex = Exp(ex_name)
ts = timeStamped('')[1:]


@ex.config
def config():
    subdir = str(Path(ex_name) / ts)
    storage_dir = str(storage_root / subdir)

    hyper_params_dir = ''

    tuning_config = load_json(Path(hyper_params_dir) / '1' / 'config.json')
    crnn_dirs = tuning_config['crnn_dirs']
    crnn_config = tuning_config['crnn_config']
    crnn_checkpoints = tuning_config['crnn_checkpoints']
    tagging_thresholds = str(Path(hyper_params_dir)/'tagging_thresholds_best_f1.json')
    del tuning_config

    dataset_name = 'desed_real_unlabel_in_domain'
    label_restriction = None

    batch_size = 64
    max_padding_rate = .1
    num_workers = 8
    device = 0

    evaluate = False
    debug = False

    ex.observers.append(FileStorageObserver.create(storage_dir))


@ex.automain
def main(
    storage_dir,
    crnn_dirs, crnn_config, crnn_checkpoints, tagging_thresholds,
    dataset_name, label_restriction, batch_size, max_padding_rate, num_workers,
    device, evaluate, debug
):

    ds = data.get_dataset(
        dataset_name, audio_reader=crnn_config['audio_reader'],
    )
    ds = data.prepare_dataset(
        ds,
        storage_dir=crnn_dirs[0],
        audio_reader=crnn_config['audio_reader'], stft=crnn_config['stft'],
        num_workers=num_workers, prefetch_buffer=4*batch_size,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=None,
        unlabeled=not evaluate,
    )

    score_mat = []
    target_mat = None
    example_ids = []
    for exp_dir, checkpoint in zip(crnn_dirs, crnn_checkpoints):
        crnn: FBCRNN = FBCRNN.from_storage_dir(
            storage_dir=exp_dir, config_name='1/config.json',
            checkpoint_name=checkpoint
        )

        crnn.to(device)
        crnn.eval()

        score_mat_i = []
        target_mat_i = []
        example_ids_i = []
        with torch.no_grad():
            for batch in ds:
                batch = example_to_device(batch, device)
                example_ids_i.extend(batch['example_id'])
                (y_fwd, y_bwd, seq_len_y), *_ = crnn(batch)
                y, _ = crnn.prediction_pooling(y_fwd, y_bwd, seq_len_y)
                score_mat_i.append(y.data.cpu().numpy())
                if evaluate:
                    assert 'events' in batch
                    target_mat_i.append(batch['events'].data.cpu().numpy())
                if debug and len(score_mat_i) > 3:
                    break
        score_mat.append(np.concatenate(score_mat_i))
        if evaluate:
            target_mat_i = np.concatenate(target_mat_i)
            if target_mat is None:
                target_mat = target_mat_i
            else:
                assert (target_mat_i == target_mat).all()
        example_ids.append(example_ids_i)
    assert all([(example_ids[i] == example_ids[0]) for i in range(len(example_ids))])
    example_ids = example_ids[0]
    score_mat = np.mean(score_mat, axis=0)
    if isinstance(tagging_thresholds, str):
        tagging_thresholds = load_json(tagging_thresholds)
    tags = score_mat > np.array(tagging_thresholds)
    if evaluate:
        assert target_mat is not None
        f, p, r = fscore(target_mat, tags, event_wise=True)
        print('F-scores:', np.round(f, decimals=3).tolist())
        print('Macro F-score:', np.round(f.mean(), decimals=3))
    labels = load_json(Path(crnn_dirs[0]) / 'events.json')
    dataset_prefix = dataset_name.split("desed_real_")[-1]
    file = Path(storage_dir) / f'{dataset_prefix}_pseudo_weak_{ts}.tsv'
    print('Output file:', file)
    with file.open('w') as fid:
        fid.write('filename\tevent_labels\n')
        for example_id, t in zip(example_ids, tags):
            event_labels = ','.join([
                labels[k] for k in np.argwhere(t > 0.5).flatten().tolist()
                if label_restriction is None
                or (isinstance(label_restriction, str) and labels[k] == label_restriction)
                or (isinstance(label_restriction, list) and labels[k] in label_restriction)
            ])
            if len(event_labels) > 0:
                fid.write(f'{example_id}.wav\t{event_labels}\n')

