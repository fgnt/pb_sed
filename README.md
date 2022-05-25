# pb_sed: Paderborn Sound Event Detection


## Installation
Install requirements:
```bash
$ pip install --user git+https://github.com/fgnt/padertorch.git@8f592c107d99cdbf8c3a5e2d7ca21373d2d3174a
$ pip install --user git+https://github.com/fgnt/paderbox.git@809b27251c478f1997d2720b89fe455aac23234e
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git@057eb86609b4928da618ede36cc745cacc4d6ba1
$ pip install --user git+https://github.com/fgnt/sed_scores_eval.git@475b83fdfcccd66508769dc14b8eb4d74740240b
```

Clone the repository:
```bash
$ git clone https://github.com/fgnt/pb_sed.git
```

Install package:
```bash
$ pip install --user -e pb_sed
```

## Database
### DESED
Install requirements:
```bash
$ pip install --user git+https://github.com/turpaultn/DESED@af3a5d5be9213239f42cf1c72f538e8058d8d2e4
```

Download the database by running
```bash
$ python -m pb_sed.database.desed.download -db /path/to/desed
```
yielding the following database structure:

```
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
```
Follow the description in https://github.com/turpaultn/DESED to request missing
files and copy them to the corresponding audio directories.

Run
```bash
$ python -m pb_sed.database.desed.create_json -db /path/to/desed
```
to create the json files ```/path/to/pb_sed/jsons/desed.json```
and ```/path/to/pb_sed/jsons/desed_pseudo_labeled.json``` (describing the database).

## Experiments
### DESED
This repository provides source code advanced from our 3-rd rank and 4-th rank
solutions for the [DCASE 2020 Challenge Task 4](http://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments-results).
and [DCASE 2021 Challenge Task 4](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments-results), respectively.
By now, however, we further improved our system performance. Changes made
compared to [[1]](#1) are
* additional time warping augmentation,
* always full overlap superposition of clips (no shifted superposition),
* reduced number of train steps in initial self-training iteration (w/o pseudo labels),
* increased number of train steps in higher self-training iteration (w/ pseudo labels),
* bigger ensembles (10 CRNNs) for pseudo labeling in each self-training iteration,
* for strong pseudo-labeling: use hyper-params giving best collar-based F-score (rather than best frame-based F-score) on validation set.
* do not mask SED scores by tag predictions for PSDS2

The resulting strongly pseudo-labeled datasets are also provided in this repo,
which allow to train a CRNN ensemble achieving >55% PSDS1, >82% PSDS2 and >63%
collar-based F1-score on the public evaluation set (when using FBCRNN ensemble
for tagging and PSDS2 and strong label CRNN ensemble for collar-based F-score
and PSDS1).

If you are using our system or our pseudo labels please consider citing this repository and our papers:

<a id="1">[1]</a> J.Ebbers and R. Haeb-Umbach,
"Self-Trained Audio Tagging and Sound Event Detection in Domestic Environments",
in Proc. Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021), 2021,

<a id="2">[2]</a> J.Ebbers and R. Haeb-Umbach,
"Forward-Backward Convolutional Recurrent Neural Networks and Tag-Conditioned Convolutional Neural Networks for Weakly Labeled Semi-Supervised Sound Event Detection",
in Proc. Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020), 2020,


#### Forward-Backward CRNN (FBCRNN)
To train an FBCRNN from scratch on only weakly labeled and synthetic data, run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training
```
Each training stores checkpoints and metadata (incl. a tensorboard event file)
in a directory ```/path/to/storage_root/weak_label_crnn_training/<group_timestamp>/<model_timestamp>```.
By default, ```/path/to/storage_root``` is ```/path/to/pb_sed/exp``` but can be
changed by setting an environment variable
```bash
$ export STORAGE_ROOT=/path/to/custom/storage_root
```

To train a second model and add it to an existing group (ensemble), run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training with group_name=<group_timestamp>
```

To train on our provided pseudo labeled data instead, add
```data_provider.json_path=/path/to/pb_sed/jsons/desed_pseudo_labeled.json```
and ```data_provider.train_set.train_unlabel_in_domain=2``` to the command, e.g.:
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training with data_provider.json_path=/path/to/pb_sed/jsons/desed_pseudo_labeled.json data_provider.train_set.train_unlabel_in_domain=2
```

For hyper-parameter tuning, run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.tuning with group_dir=/path/to/storage_root/weak_label_crnn_training/<group_timestamp>
```
which saves hyper-parameters in a directory ```/path/to/storage_root/weak_label_crnn_hyper_params/<timestamp>```.

For evaluation on the public evaluation set, run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.inference with hyper_params_dir=/path/to/storage_root/weak_label_crnn_hyper_params/<timestamp>
```


#### Strong label CRNN
To train a CRNN with our provided strong pseudo labels, run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.training with weak_label_crnn_hyper_params_dir=/path/to/storage_root/weak_label_crnn_hyper_params/<timestamp>
```
Each training stores checkpoints and metadata (incl. a tensorboard event file)
in a directory ```/path/to/storage_root/strong_label_crnn_training/<group_timestamp>/<model_timestamp>```.

To train a second model and add it to an existing group (ensemble), run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.training with weak_label_crnn_hyper_params_dir=/path/to/storage_root/weak_label_crnn_hyper_params/<timestamp> group_name=<group_timestamp>
```

For hyper-parameter tuning, run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.tuning with strong_label_crnn_group_dir=/path/to/storage_root/strong_label_crnn_training/<group_timestamp> weak_label_crnn_hyper_params_dir=/path/to/storage_root/weak_label_crnn_hyper_params/<timestamp>
```
which saves hyper-parameters in a directory ```/path/to/storage_root/strong_label_crnn_hyper_params/<timestamp>```.

For evaluation on the public evaluation set, run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.inference with strong_label_crnn_hyper_params_dir=/path/to/storage_root/strong_label_crnn_hyper_params/<timestamp>
```
