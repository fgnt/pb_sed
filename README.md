# pb_sed: Paderborn Sound Event Detection

This repository provides the source code for our 1-st rank solution for
[DCASE 2022 Challenge Task 4](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments-results),
which advanced from our 3-rd rank and 4-th rank solutions for the
[DCASE 2020 Challenge Task 4](https://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments-results)
and [DCASE 2021 Challenge Task 4](https://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments-results),
respectively.

This repository also provides our final strongly pseudo-labeled datasets
* [without using external data](https://github.com/fgnt/pb_sed/tree/master/exp/strong_label_crnn_inference/2022-05-04-09-05-53): allow to train a CRNN ensemble
  achieving >55% PSDS1, >82% PSDS2 and >65% collar-based F1-score on the public
  evaluation set (when using FBCRNN ensemble for tagging and PSDS2,
  tag-conditioned BiCRNN ensemble for collar-based F1-score and unconditioned
  BiCRNN ensemble for PSDS1).
* [with using external data](https://github.com/fgnt/pb_sed/tree/master/exp/strong_label_crnn_inference/2022-06-24-10-06-21): allow to train a CRNN ensemble
  achieving >58% PSDS1, >86% PSDS2 and >70% collar-based F1-score on the public
  evaluation set (when using FBCRNN ensemble for tagging and PSDS2,
  tag-conditioned BiCRNN ensemble for collar-based F1-score and unconditioned
  BiCRNN ensemble for PSDS1).

If you are using our system or our pseudo labels please consider citing our papers:

<a id="2">[1]</a> J.Ebbers and R. Haeb-Umbach,
"Pre-Training and Self-Training for Sound Event Detection in Domestic Environments",
Technical Report for Challenge on Detection and Classification of Acoustic Scenes and Events 2022,

<a id="1">[2]</a> J.Ebbers and R. Haeb-Umbach,
"Self-Trained Audio Tagging and Sound Event Detection in Domestic Environments",
in Proc. Workshop on Detection and Classification of Acoustic Scenes and Events 2021,

<a id="2">[3]</a> J.Ebbers and R. Haeb-Umbach,
"Forward-Backward Convolutional Recurrent Neural Networks and Tag-Conditioned Convolutional Neural Networks for Weakly Labeled Semi-Supervised Sound Event Detection",
in Proc. Workshop on Detection and Classification of Acoustic Scenes and Events 2020,


## Installation
Install requirements:
```bash
$ pip install --user git+https://github.com/fgnt/padertorch.git@b7ba24a42a05745d127a74a519af08a876319a95
$ pip install --user git+https://github.com/fgnt/paderbox.git@809b27251c478f1997d2720b89fe455aac23234e
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git@dc9f487bd433a9ccc8e157d58e338074e3cd8705
$ pip install --user git+https://github.com/fgnt/sed_scores_eval.git@a922e0a4692957d56b307a2eec942422ab22b55a
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
│   │       └── <clip_id>.wav
│   ├── train
│   │   ├── strong
│   │   │   └── <clip_id>.wav
│   │   ├── synthetic20
│   │   │   └── <clip_id>.wav
│   │   ├── synthetic21
│   │   │   └── <clip_id>.wav
│   │   ├── unlabel_in_domain
│   │   │   └── <clip_id>.wav
│   │   └── weak
│   │       └── <clip_id>.wav
│   └── validation
│       └── validation
│           └── <clip_id>.wav
├── metadata
│   ├── eval
│   │   └── public.tsv
│   ├── train
│   │   ├── strong.tsv
│   │   ├── synthetic20.tsv
│   │   ├── synthetic21.tsv
│   │   ├── unlabel_in_domain.tsv
│   │   └── weak.tsv
│   └── validation
│       └── validation.tsv
└── missing_files
    ├── missing_files_strong.tsv
    ├── missing_files_unlabel_in_domain.tsv
    ├── missing_files_validation.tsv
    └── missing_files_weak.tsv
```
Follow the description in https://github.com/turpaultn/DESED to request missing
files and copy them to the corresponding audio directories.

Run
```bash
$ python -m pb_sed.database.desed.create_json -db /path/to/desed
```
to create the json files ```/path/to/pb_sed/jsons/desed.json```, 
```/path/to/pb_sed/jsons/desed_pseudo_labeled_without_external.json``` and 
```/path/to/pb_sed/jsons/desed_pseudo_labeled_with_external.json``` (describing the database).


### AudioSet
To download the whole AudioSet run
```bash
$ python -m pb_sed.database.audioset.download -db /path/to/audioset
```
yielding the following database structure:

```
├── audio
│   ├── balanced_train
│   │   └── <clip_id>.wav
│   ├── eval
│   │   └── <clip_id>.wav
│   └── unbalanced_train
│       └── <clip_id>.wav
├── audioset_eval_strong.tsv
├── audioset_train_strong.tsv
├── balanced_train_segments.csv
├── class_labels_indices.csv
├── eval_segments.csv
├── mid_to_display_name.tsv
└── unbalanced_train_segments.csv
```

Note, that this can take multiple days as AudioSet is huge.
You may prefer to setup above database structure with symlinks towards your
existing AudioSet download.

Run
```bash
$ python -m pb_sed.database.audioset.create_json -db /path/to/audioset
```
to create the json file ```/path/to/pb_sed/jsons/audioset.json``` (describing the database).


## Experiments
### Forward-Backward CRNN (FBCRNN)
To train an FBCRNN from scratch, run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training
```
Each training stores checkpoints and metadata (incl. a tensorboard event file)
in a directory ```/path/to/storage_root/weak_label_crnn/desed/training/<group_timestamp>/<model_timestamp>```.
By default, ```/path/to/storage_root``` is ```/path/to/pb_sed/exp``` but can be
changed by setting an environment variable
```bash
$ export STORAGE_ROOT=/path/to/custom/storage_root
```

To train a second model and add it to an existing group (ensemble), run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training with group_name=<group_timestamp>
```

To train on our provided pseudo labeled data, add
```data_provider.json_path=/path/to/pb_sed/jsons/desed_pseudo_labeled_{with,without}_external.json```
and ```data_provider.train_set.train_unlabel_in_domain=2``` to the command, e.g.:
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training with data_provider.json_path=/path/to/pb_sed/jsons/desed_pseudo_labeled_with_external.json data_provider.train_set.train_unlabel_in_domain=2
```

Add ```external_data=False``` to the commands to exclude external data from FBCRNN training.
Add ```batch_size=<batch size>``` to the commands to adjust the batch size (e.g. if CUDA out of memory).

For hyper-parameter tuning, run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.tuning with group_dir=/path/to/storage_root/weak_label_crnn/desed/training/<group_timestamp>
```
which saves hyper-parameters in a directory ```/path/to/storage_root/weak_label_crnn/desed/hyper_params/<timestamp>```.

For evaluation on the public evaluation set, run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.inference with hyper_params_dir=/path/to/storage_root/weak_label_crnn/desed/hyper_params/<timestamp>
```

To perform pseudo labeling, run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.inference with hyper_params_dir=/path/to/storage_root/weak_label_crnn/desed/hyper_params/<timestamp> dataset_name='["train_weak","train_unlabel_in_domain"]' weak_pseudo_labeling='[False,True]' boundary_pseudo_labeling=True
```
which will write a file ```/path/to/storage_root/weak_label_crnn/desed/inference/<timestamp>/desed.json``` with pseudo labeled data.

To train on this pseudo labeled data, add (similar to training on our provided pseudo labeled data)
```data_provider.json_path=/path/to/storage_root/weak_label_crnn/desed/inference/<timestamp>/desed.json```
and ```data_provider.train_set.train_unlabel_in_domain=2``` to a training command.


### Bidirectional CRNN (requiring strong labels)
To train an unconditioned bidirectional CRNN (BiCRNN) with our provided strong pseudo labels (with external data), run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.training with weak_label_crnn_hyper_params_dir=/path/to/storage_root/weak_label_crnn/desed/hyper_params/<timestamp>
```
Each training stores checkpoints and metadata (incl. a tensorboard event file)
in a directory ```/path/to/storage_root/strong_label_crnn/desed/training/<group_timestamp>/<model_timestamp>```.

To train a second model and add it to an existing group (ensemble), run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.training with weak_label_crnn_hyper_params_dir=/path/to/storage_root/weak_label_crnn/desed/hyper_params/<timestamp> group_name=<group_timestamp>
```

To train tag-conditioned BiCRNNs instead add ```trainer.model.tag_conditioning=True``` to the commands.

Add ```external_data=False``` to the commands to exclude external data from BiCRNN training and to use pseudo labels obtained without external data.

For hyper-parameter tuning, run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.tuning with strong_label_crnn_group_dir=/path/to/storage_root/strong_label_crnn/desed/training/<group_timestamp> weak_label_crnn_hyper_params_dir=/path/to/storage_root/weak_label_crnn/desed/hyper_params/<timestamp>
```
which saves hyper-parameters in a directory ```/path/to/storage_root/strong_label_crnn/desed/hyper_params/<timestamp>```.

For evaluation on the public evaluation set, run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.inference with strong_label_crnn_hyper_params_dir=/path/to/storage_root/strong_label_crnn/desed/hyper_params/<timestamp>
```

To perform pseudo labeling, run
```bash
$ python -m pb_sed.experiments.strong_label_crnn.inference with strong_label_crnn_hyper_params_dir=/scratch/hpc-prf-nt1/ebbers/exp/strong_label_crnn_hyper_params/2022-06-13-11-15-54 dataset_name='["train_weak","train_unlabel_in_domain"]' strong_pseudo_labeling=True
```
which will write a file ```/path/to/storage_root/strong_label_crnn/desed/inference/<timestamp>/desed.json``` with pseudo labeled data.

To train on this pseudo labeled data (instead of our provided pseudo labeled data), add
```data_provider.json_path=/path/to/storage_root/strong_label_crnn/desed/inference/<timestamp>/desed.json``` to a training command.


### AudioSet Pre-training
To pre-train a deeper and wider FBCRNN on AudioSet (excluding DESED validation clips), run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training with database=audioset net_config=deep width=2 filter_desed_test_clips=True
```

To train an FBCRNN from the pretrained model (with some frozen layers), run
```bash
$ python -m pb_sed.experiments.weak_label_crnn.training with net_config=deep width=2 init_ckpt_path=/path/to/storage_root/weak_label_crnn/audioset/training/<group_timestamp>/<model_timestamp> frozen_cnn_2d_layers=18 frozen_cnn_1d_layers=1
```

To train an unconditioned BiCRNN from the pretrained model (with some frozen layers), run
```bash
$ python -m pb_sed.experiments.strong_crnn.training with net_config=deep width=2 init_ckpt_path=/path/to/storage_root/weak_label_crnn/audioset/training/<group_timestamp>/<model_timestamp> frozen_cnn_2d_layers=18 frozen_cnn_1d_layers=1 weak_label_crnn_hyper_params_dir=/path/to/storage_root/weak_label_crnn/desed/hyper_params/<timestamp>
```

To train a tag-conditioned BiCRNN instead, add ```trainer.model.tag_conditioning=True``` to the command.
