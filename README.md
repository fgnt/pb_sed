# pb_sed
Paderborn Sound Event Detection


## Installation
Install requirements
```bash
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git@d500d23d23c0cc2ebb874c4974b4ffa7a2418b96
$ pip install --user git+https://github.com/fgnt/paderbox.git@79bb3eadaa944c3ae8e3a4f411469f35e5e177c5
$ pip install --user git+https://github.com/fgnt/padertorch.git@34a0a95ec696cb8b6f24c62ed500895b352c52ab
```

Clone the repo
```bash
$ git clone https://github.com/fgnt/pb_sed.git
```

Install this package
```bash
$ pip install --user -e pb_sed
```

## Database
### DESED (DCASE 2020 Task 4)
Install requirements
```bash
$ pip install --user git+https://github.com/turpaultn/DESED@2fb7fe0b4b33569ad3693d09e50037b8b4206b72
```

Download
```bash
$ python -m pb_sed.database.desed.download -db /path/to/desed
```

yielding database structure

```
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

```

Create json file (describing the database)
```bash
$ python -m pb_sed.database.desed.create_json -db /path/to/desed
```

## Experiments
### DCASE 2020 Task 4

Train FBCRNN
```bash
$ python -m pb_sed.experiments.dcase_2020_task_4.train_fbcrnn
```

Train tag conditioned CNN
```bash
$ python -m pb_sed.experiments.dcase_2020_task_4.train_tag_conditioned_cnn
```