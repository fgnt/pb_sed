from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from paderbox.io.json_module import load_json, dump_json
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.data.transforms import Collate
from padertorch.data import example_to_device
from pb_sed.evaluation import instance_based, event_based
from pb_sed.experiments.dcase_2020_task_4 import data
from pb_sed.models.crnn import CRNN
from pb_sed.models.cnn import CNN
from pb_sed.paths import storage_root
from pb_sed.utils import medfilt
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver

ex_name = 'dcase_2020_hyper_params'
ex = Exp(ex_name)


@ex.config
def config():
    subdir = str(Path(ex_name) / timeStamped('')[1:])
    storage_dir = str(storage_root / subdir)

    crnn_dirs = []
    assert len(crnn_dirs) > 0, 'Set crnn_dirs on the command line.'
    crnn_config = load_json(Path(crnn_dirs[0]) / '1' / 'config.json')
    crnn_checkpoints = len(crnn_dirs) * ['ckpt_best_mean_fscore.pth']
    cnn_dirs = []
    cnn_config = None if len(cnn_dirs) == 0 else load_json(Path(cnn_dirs[0]) / '1' / 'config.json')
    cnn_checkpoints = len(cnn_dirs) * ['ckpt_best_mean_fscore.pth']

    batch_size = 16
    max_padding_rate = .1
    num_workers = 8
    device = 0

    contexts = [20, 15, 10, 5]
    ensembles = ['hybrid', 'cnn', 'crnn']
    medfilt_sizes = [51, 41, 31, 21, 11]

    event_based_collar = 10
    event_based_offset_collar_rate = .2

    print_decimals = 3

    debug = False

    ex.observers.append(FileStorageObserver.create(storage_dir))


@ex.automain
def main(
    storage_dir,
    crnn_dirs, crnn_config, crnn_checkpoints, cnn_dirs, cnn_checkpoints,
    batch_size, max_padding_rate,  num_workers, device,
    contexts, medfilt_sizes, ensembles,
    event_based_collar, event_based_offset_collar_rate,
    print_decimals, debug
):
    assert all([ensemble in ['hybrid', 'crnn', 'cnn'] for ensemble in ensembles]), ensembles
    validation_set = data.get_dataset(
        'desed_real_validation', audio_reader=crnn_config['audio_reader'],
    )
    validation_iter = data.prepare_dataset(
        validation_set,
        storage_dir=crnn_dirs[0],
        audio_reader=crnn_config['audio_reader'], stft=crnn_config['stft'],
        num_workers=num_workers, prefetch_buffer=4 * batch_size,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=None,
        add_alignment=True,
    )

    crnns = []

    # tagging:
    print('Audio Tagging:')
    tagging_score_mat = []
    tagging_target_mat = None
    for exp_dir, crnn_checkpoint in zip(crnn_dirs, crnn_checkpoints):
        crnn: CRNN = CRNN.from_storage_dir(
            storage_dir=exp_dir, config_name='1/config.json',
            checkpoint_name=crnn_checkpoint
        )
        crnn.to(device)
        crnn.eval()

        score_mat_i = []
        target_mat_i = []
        n = 0
        with torch.no_grad():
            for batch in validation_iter:
                batch = example_to_device(batch, device)
                (y_fwd, y_bwd, seq_len_y), *_ = crnn(batch)
                y, _ = crnn.prediction_pooling(y_fwd, y_bwd, seq_len_y)
                score_mat_i.append(y.data.cpu().numpy())
                target_mat_i.append(batch['events'].data.cpu().numpy())
                n += y.shape[0]
                if debug and n >= 64:
                    break
        crnn.to('cpu')
        crnns.append(crnn)
        tagging_score_mat.append(np.concatenate(score_mat_i))
        target_mat_i = np.concatenate(target_mat_i)
        if tagging_target_mat is None:
            tagging_target_mat = target_mat_i
        else:
            assert (target_mat_i == tagging_target_mat).all()
    tagging_score_mat = np.mean(tagging_score_mat, axis=0)
    th_f, best_f = instance_based.get_optimal_thresholds(tagging_target_mat, tagging_score_mat, metric='f1')
    dump_json(th_f.tolist(), Path(storage_dir) / 'tagging_thresholds_best_f1.json')
    print('  F-scores:', np.round(best_f, decimals=print_decimals).tolist())
    print('  Macro F-score:', np.round(best_f.mean(), decimals=print_decimals))
    th_er, best_er = instance_based.get_optimal_thresholds(tagging_target_mat, tagging_score_mat, metric='er')
    dump_json(th_er.tolist(), Path(storage_dir) / 'tagging_thresholds_best_er.json')
    print('  Error-rates:', np.round(best_er, decimals=print_decimals).tolist())
    print('  Macro error-rate:', np.round(best_er.mean(), decimals=print_decimals))

    tags = tagging_score_mat > th_f
    del tagging_score_mat
    del tagging_target_mat

    # detection:
    print('Sound event detection:')
    best_frame_er = defaultdict(list)
    best_frame_f = defaultdict(list)
    best_event_f = defaultdict(list)

    def sweep_medfilts(target_mat, score_mat, name):
        ensemble = name.split('_')[0]
        for medfilt_size in medfilt_sizes:
            print(f'    Medfilt size: {medfilt_size}')
            if medfilt_size > 1:
                score_mat_filtered = medfilt(
                    score_mat.astype(np.float), medfilt_size, axis=1
                )
            else:
                assert medfilt_size == 1, medfilt_size
                score_mat_filtered = score_mat
            thres, f1 = instance_based.get_optimal_thresholds(
                rearrange(target_mat, 'b t k -> (b t) k'),
                rearrange(score_mat_filtered, 'b t k -> (b t) k'),
                metric='f1'
            )
            dump_json(
                thres.tolist(),
                Path(storage_dir) /
                f'detection_thresholds_best_frame_f1_{name}_{medfilt_size}.json'
            )
            print('      Frame-based F-scores:', np.round(f1, decimals=print_decimals).tolist())
            print('      Frame-based macro F-score', np.round(f1.mean(), decimals=print_decimals))
            for i, f_i in enumerate(f1):
                if len(best_frame_f[ensemble]) <= i:
                    assert len(best_frame_f[ensemble]) == i, (
                        i, len(best_frame_f[ensemble])
                    )
                    best_frame_f[ensemble].append((f_i, f'{name}_{medfilt_size}'))
                elif f_i > best_frame_f[ensemble][i][0]:
                    best_frame_f[ensemble][i] = (f_i, f'{name}_{medfilt_size}')

            thres, er = instance_based.get_optimal_thresholds(
                rearrange(target_mat, 'b t k -> (b t) k'),
                rearrange(score_mat_filtered, 'b t k -> (b t) k'),
                metric='er'
            )

            dump_json(
                thres.tolist(),
                Path(storage_dir) /
                f'detection_thresholds_best_frame_er_{name}_{medfilt_size}.json'
            )
            print('      Frame-based error-rates:', np.round(er, decimals=print_decimals).tolist())
            print('      Frame-based macro error-rate:', np.round(er.mean(), decimals=print_decimals))
            for i, er_i in enumerate(er):
                if len(best_frame_er[ensemble]) <= i:
                    assert len(best_frame_er[ensemble]) == i, (
                    i, len(best_frame_er[ensemble]))
                    best_frame_er[ensemble].append((er_i, f'{name}_{medfilt_size}'))
                elif er_i < best_frame_er[ensemble][i][0]:
                    best_frame_er[ensemble][i] = (er_i, f'{name}_{medfilt_size}')

            thres, f1 = event_based.get_optimal_thresholds(
                target_mat, score_mat_filtered, metric='f1',
                collar=event_based_collar,
                offset_collar_rate=event_based_offset_collar_rate,
            )

            dump_json(
                thres.tolist(),
                Path(storage_dir) /
                f'detection_thresholds_best_event_f1_{name}_{medfilt_size}.json'
            )
            print('      Event-based F-scores:', np.round(f1, decimals=print_decimals).tolist())
            print('      Event-based macro F-score:', np.round(f1.mean(), decimals=print_decimals))
            for i, f_i in enumerate(f1):
                if len(best_event_f[ensemble]) <= i:
                    assert len(best_event_f[ensemble]) == i, (
                        i, len(best_event_f[ensemble])
                    )
                    best_event_f[ensemble].append((f_i, f'{name}_{medfilt_size}'))
                elif f_i > best_event_f[ensemble][i][0]:
                    best_event_f[ensemble][i] = (f_i, f'{name}_{medfilt_size}')

    detection_target_mat = None
    cnn_detection_score_mat = None
    if any([ensemble != 'crnn' for ensemble in ensembles]):
        assert len(cnn_dirs) > 0
        cnn_detection_score_mat = []
        for exp_dir, cnn_checkpoint in zip(cnn_dirs, cnn_checkpoints):
            cnn: CNN = CNN.from_storage_dir(
                storage_dir=exp_dir, config_name='1/config.json',
                checkpoint_name=cnn_checkpoint
            )

            cnn.to(device)
            cnn.eval()

            score_mat_i = []
            target_mat_i = []
            i = 0
            with torch.no_grad():
                for batch in validation_iter:
                    batch = example_to_device(batch, device)
                    x = batch['stft']
                    b = x.shape[0]
                    seq_len = batch['seq_len']
                    targets = batch['events_alignment']
                    (y, seq_len), _ = cnn.predict(
                        x,
                        torch.Tensor(tags[i:i+b].astype(np.float32)).to(x.device),
                        seq_len
                    )
                    y = y.data.cpu().numpy() * tags[i:i + b, :, None]
                    score_mat_i.extend(
                        [y[j, :, :l].T for j, l in enumerate(seq_len)]
                    )
                    target_mat_i.extend(
                        [targets[j, :, :l].data.cpu().numpy().T for j, l in
                         enumerate(seq_len)]
                    )
                    if debug and len(score_mat_i) >= 64:
                        break
                    i += b
            cnn.to('cpu')
            cnn_detection_score_mat.append(Collate()(score_mat_i))
            target_mat_i = Collate()(target_mat_i)
            if detection_target_mat is None:
                detection_target_mat = target_mat_i
            else:
                assert (target_mat_i == detection_target_mat).all()
        cnn_detection_score_mat = np.mean(cnn_detection_score_mat, axis=0)

        if 'cnn' in ensembles:
            print(f'  Ensemble: cnn')
            sweep_medfilts(detection_target_mat, cnn_detection_score_mat, 'cnn')

    if any([ensemble != 'cnn' for ensemble in ensembles]):
        for context in contexts:
            crnn_detection_score_mat = []
            for crnn in crnns:
                crnn.to(device)
                crnn.eval()

                score_mat_i = []
                target_mat_i = []
                i = 0
                with torch.no_grad():
                    for batch in validation_iter:
                        batch = example_to_device(batch, device)
                        x = batch['stft']
                        seq_len = batch['seq_len']
                        targets = batch['events_alignment'].transpose(1, 2).detach().cpu().numpy()
                        y_sed, seq_len = crnn.sed(x, context, seq_len)
                        b, t, _ = y_sed.shape
                        y_sed = y_sed.detach().cpu().numpy() * tags[i:i+b, None]
                        r = np.ceil(x.shape[2]/t)
                        y_sed = np.repeat(y_sed, r, axis=1)
                        score_mat_i.extend([y_sed[j, :l, :] for j, l in enumerate(seq_len)])
                        target_mat_i.extend([targets[j, :, :l] for j, l in enumerate(seq_len)])
                        if debug and len(score_mat_i) >= 64:
                            break
                        i += b
                crnn.to('cpu')
                crnn_detection_score_mat.append(Collate()(score_mat_i))
                target_mat_i = Collate()(target_mat_i)
                if detection_target_mat is None:
                    detection_target_mat = target_mat_i
                else:
                    assert (target_mat_i == detection_target_mat).all()
            crnn_detection_score_mat = np.mean(crnn_detection_score_mat, axis=0)

            if 'hybrid' in ensembles:
                name = f'hybrid_{context}'
                print(f'  Ensemble: {name}')
                assert cnn_detection_score_mat is not None
                mean_detection_score_mat = (crnn_detection_score_mat + cnn_detection_score_mat)/2
                sweep_medfilts(
                    detection_target_mat, mean_detection_score_mat,
                    f'hybrid_{context}'
                )
            if 'crnn' in ensembles:
                name = f'crnn_{context}'
                print(f'  Ensemble: {name}')
                sweep_medfilts(
                    detection_target_mat, crnn_detection_score_mat,
                    f'crnn_{context}'
                )

    best_final_frame_f = None
    best_final_frame_er = None
    best_final_event_f = None
    for ensemble in ensembles:
        if best_final_frame_f is None:
            best_final_frame_f = best_frame_f[ensemble]
            best_final_frame_er = best_frame_er[ensemble]
            best_final_event_f = best_event_f[ensemble]
        else:
            best_final_frame_f = [
                opponent if opponent[0] > cur_best[0] else cur_best
                for cur_best, opponent in zip(
                    best_final_frame_f, best_frame_f[ensemble]
                )
            ]
            best_final_frame_er = [
                opponent if opponent[0] > cur_best[0] else cur_best
                for cur_best, opponent in zip(
                    best_final_frame_er, best_frame_er[ensemble]
                )
            ]
            best_final_event_f = [
                opponent if opponent[0] > cur_best[0] else cur_best
                for cur_best, opponent in zip(
                    best_final_event_f, best_event_f[ensemble]
                )
            ]
        print('\n')
        best_config = [entry[1] for entry in best_frame_f[ensemble]]
        dump_json(best_config, Path(storage_dir) / f'best_frame_f1_{ensemble}_config.json')
        best_f = [entry[0] for entry in best_frame_f[ensemble]]
        print(f'Best frame-based {ensemble} F-score configuration:', best_config)
        print(f'Best frame-based {ensemble} F-scores:', np.round(best_f, decimals=print_decimals).tolist())
        print(f'Best frame-based {ensemble} macro F-score:', np.round(np.mean(best_f), decimals=print_decimals))
        print('')
        best_config = [entry[1] for entry in best_frame_er[ensemble]]
        dump_json(best_config, Path(storage_dir) / f'best_frame_er_{ensemble}_config.json')
        best_er = [entry[0] for entry in best_frame_er[ensemble]]
        print(f'Best frame-based {ensemble} error-rate configuration:', best_config)
        print(f'Best frame-based {ensemble} error-rates:', np.round(best_er, decimals=print_decimals).tolist())
        print(f'Best frame-based {ensemble} macro error-rate:', np.round(np.mean(best_er), decimals=print_decimals))
        print('')
        best_config = [entry[1] for entry in best_event_f[ensemble]]
        dump_json(best_config, Path(storage_dir) / f'best_event_f1_{ensemble}_config.json')
        best_f = [entry[0] for entry in best_event_f[ensemble]]
        print(f'Best event-based {ensemble} F-score configuration:', best_config)
        print(f'Best event-based {ensemble} F-scores:', np.round(best_f, decimals=print_decimals).tolist())
        print(f'Best event-based {ensemble} macro F-score:', np.round(np.mean(best_f), decimals=print_decimals))
        print('')

    print('\n')
    best_config = [entry[1] for entry in best_final_frame_f]
    dump_json(best_config, Path(storage_dir) / f'best_frame_f1_final_config.json')
    best_f = [entry[0] for entry in best_final_frame_f]
    print('Best frame-based final F-score configuration:', best_config)
    print('Best frame-based final F-scores:', np.round(best_f, decimals=print_decimals).tolist())
    print('Best frame-based final macro F-score:', np.round(np.mean(best_f), decimals=print_decimals))
    print('')
    best_config = [entry[1] for entry in best_final_frame_er]
    dump_json(best_config, Path(storage_dir) / f'best_frame_er_final_config.json')
    best_er = [entry[0] for entry in best_final_frame_er]
    print('Best frame-based final error-rate configuration:', best_config)
    print('Best frame-based final error-rates:', np.round(best_er, decimals=print_decimals).tolist())
    print('Best frame-based final macro error-rate:', np.round(np.mean(best_er), decimals=print_decimals))
    print('')
    best_config = [entry[1] for entry in best_final_event_f]
    dump_json(best_config, Path(storage_dir) / f'best_event_f1_final_config.json')
    best_f = [entry[0] for entry in best_final_event_f]
    print('Best event-based final F-score configuration:', best_config)
    print('Best event-based final F-scores:', np.round(best_f, decimals=print_decimals).tolist())
    print('Best event-based final macro F-score:', np.round(np.mean(best_f), decimals=print_decimals))
    print('\n')
    print('Storage directory:', storage_dir)
