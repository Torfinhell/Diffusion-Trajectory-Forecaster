# Diffusion-Trajectory-Forecaster
Train Diffusion Trajectory Forecaster model using the Waymo Motion dataset  for  mmls hse project

### Install Dependencies
To create env and install dependecies:
```
conda create -n diffusion_tracker python=3.10
conda activate diffusion_tracker
pip install uv
uv sync
```

### Docker
You can run the whole project inside Docker with GPU access instead of installing the Python environment locally.

1. Build the image from the repository root:
```bash
docker build -t diffusion-trajectory-forecaster .
```

2. Make the helper script executable:
```bash
chmod +x scripts/docker_run.sh
```

3. Start an interactive shell inside the container:
```bash
scripts/docker_run.sh bash
```

4. Run project commands inside that shell:
```bash
uv run python train.py
uv run python train.py data=processed_v2
uv run python create_dataset.py data=processed_v1
uv run pytest
uv run python visualise_data.py
uv run dvc pull
uv run dvc push
```

How it works:
- the repository is mounted into the container at `/app`
- your code, checkpoints, outputs, and local changes stay on the host machine
- the container uses its own virtual environment at `/opt/venv`, so Docker does not recreate or modify your host `.venv`
- the helper script runs the container with your host UID/GID so generated files remain writable by your user and Git can stage them
- container-side cache and auth files are stored in gitignored `.docker-cache/`
- DVC secrets should stay in `.dvc/config.local`, not `.dvc/config`


Notes:
- rebuild the image after this change so the container environment is created under `/opt/venv`

To authenticate to google account for data downloading(one time):

1. Apply for [Waymo Open Dataset](https://waymo.com/open) access.
2. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
3. Run ```gcloud auth login <your_email>``` with the same email used for step 1.
4. Run ```gcloud auth application-default login```.

### DVC setup
Processed datasets are tracked with DVC as directory artifacts. Git stores the `.dvc` metadata files, while the actual `.pkl` files live locally or in the configured DVC remote.

Remote configuration:
- keep the remote URL in `.dvc/config`
- keep credentials such as `access_key_id` and `secret_access_key` in `.dvc/config.local`
- do not commit `.dvc/config.local`

Amazon S3 credentials setup:
```bash
uv run dvc remote list
uv run dvc remote modify --local myremote access_key_id <AWS_ACCESS_KEY_ID>
uv run dvc remote modify --local myremote secret_access_key <AWS_SECRET_ACCESS_KEY>
```

Notes:
- the shared repository config already defines the default DVC remote URL and region
- the AWS identity used by DVC should have `s3:ListBucket`, `s3:GetObject`, `s3:PutObject`, and `s3:DeleteObject`
- if the bucket uses SSE-KMS encryption, the same identity also needs the matching KMS permissions
- if AWS CLI is already configured on the machine, DVC can also reuse that configuration

Pull datasets on a new machine:
```bash
uv run dvc pull
```

Pull one dataset explicitly:
```bash
uv run dvc pull data/processed_v1.dvc
uv run dvc pull data/processed_v2.dvc
uv run dvc pull data/baseline1.dvc
```

Push updated artifacts:
```bash
uv run dvc push
```


Useful checks:
```bash
uv run dvc status
uv run dvc list . data
```

### Dataset creation
To build processed train/val/test datasets from raw Waymo data:
```bash
uv run python create_dataset.py data=processed_v1 dataset_creation=default
```

This script:
- creates the processed `.pkl` files inside `data/processed_v1/`
- runs `dvc add` on the dataset directory once
- stages the generated `.dvc` file in Git
- runs `dvc push` if a working DVC remote is configured

Main configs:
- dataset artifact paths: `src/configs/data/processed_v1.yaml`
- dataset creation settings: `src/configs/dataset_creation/default.yaml`

When creating a new dataset version, set both config groups:
```bash
uv run python create_dataset.py data=baseline1 dataset_creation=baseline1
```

Why both are needed:
- `data=...` chooses where the dataset is saved and which `.dvc` file is updated
- `dataset_creation=...` chooses how the dataset is generated

If you reuse the same `data=...` name, the local files and the corresponding `.dvc` artifact are updated to the new content.

You can still override individual settings with Hydra, for example:
```bash
uv run python create_dataset.py dataset_creation.train.num_states=100
uv run python create_dataset.py dataset_creation.val.max_num_objects=16
```

### Dataset formats
The repository currently supports two processed dataset formats:

1. Legacy chunked pickle format
2. WebDataset shard format

Legacy chunked pickle format:
- storage format value: `directory_chunks`
- output layout:
  - one manifest file such as `train_processed_v1.pkl`
  - one chunk directory such as `train_processed_v1.pkl.chunks/`
- training path:
  - `src/data_module/legacy_dataset.py`
  - custom chunk cache
  - custom chunk sampler from `src/data_module/sampler.py`

WebDataset shard format:
- storage format value: `webdataset`
- output layout:
  - one shard directory such as `train_processed_v1.pkl.wds/`
  - many tar shards inside it, for example `shard-000000.tar`
  - one `index.json`
- training path:
  - `src/data_module/dataset.py`
  - standard `webdataset.WebDataset(...)`
  - no custom chunk sampler


Create the legacy chunked format:
```bash
uv run python create_dataset.py \
  data=small_base_wo_vis \
  dataset_creation=small_base_wo_vis \
  storage_format=directory_chunks
```

Create the WebDataset format:
```bash
uv run python create_dataset.py \
  data=small_base_wo_vis \
  dataset_creation=small_base_wo_vis \
  storage_format=webdataset
```

Train with the legacy chunked format:
```bash
uv run python train.py \
  data=small_base_wo_vis \
  data.train.storage_format=directory_chunks \
  data.val.storage_format=directory_chunks \
  data.test.storage_format=directory_chunks
```

Train with the WebDataset format:
```bash
uv run python train.py \
  data=small_base_wo_vis \
  data.train.storage_format=webdataset \
  data.val.storage_format=webdataset \
  data.test.storage_format=webdataset
```

How the training path is selected:
- `src/configs/data/*.yaml` contains `storage_format` for each split
- if `storage_format=webdataset`, training uses the WebDataset loader
- otherwise it falls back to the legacy chunked dataset loader

### Config guide
The project uses Hydra configs from `src/configs/`. The script you run decides which config groups are read.

`train.py`:
- root config: `src/configs/ddpm_baseline.yaml`
- alternative training preset: `src/configs/ddpm_1.yaml`
- config groups used by training:
- `model=...`: model class and model-specific hyperparameters such as architecture size, diffusion settings, learning rate, oracle settings, and checkpoint loading
- `data=...`: which processed dataset artifact is used for train/val/test, including the `.pkl` paths and matching `.dvc` file
- `dataloaders=...`: batch size, shuffle, workers, and other `DataLoader` settings
- `logger=...`: experiment logger backend and ClearML project/run settings
- `metrics=...`: train/validation metrics instantiated during training
- `visual=...`: visualization and sampling/debug rendering settings used by the model during validation/logging
- `trainer.*`: top-level training loop settings such as epochs, train/val epoch length, gradient clipping, seed, logging mode, and checkpoint reload flag

ClearML setup:
- initialize ClearML credentials once with `uv run clearml-init`
- default logger config: `src/configs/logger/clearml.yaml`
- `logger.project_name`: shared ClearML project for a family of experiments
- `logger.task_name`: individual run name shown in ClearML
- recommended pattern: keep one fixed `logger.project_name` and override `logger.task_name` per run

Examples:
```bash
uv run clearml-init
uv run python train.py logger.project_name=my_experiments logger.task_name=exp_001
uv run python train.py logger.project_name=my_experiments logger.task_name=attn_v2_lr1e-4
uv run python train.py logger.task_name=baseline_processed_v2
```

Examples:
```bash
uv run python train.py
uv run python train.py data=processed_v2
uv run python train.py model=diffusion_attn_2x trainer.num_epochs=230
```

### Usual vs debug models
The project now has separate normal and debug model classes.

Usual attention model:
```bash
uv run python train.py model=diffusion_attn
```

Debug attention model:
```bash
uv run python train.py model=diffusion_attn_debug
```

You can still override individual debug flags when using the debug model, for example:
```bash
uv run python train.py model=diffusion_attn_debug visual.debug_metrics=true
uv run python train.py model=diffusion_attn_debug visual.debug_denoiser_scale=true
uv run python train.py model=diffusion_attn_debug model.oracle_cfg.use_for_sampling=true
```

How model selection works:
- `src/configs/model/diffusion_attn.yaml` uses `src.models.DiffusionAttentionModel`
- `src/configs/model/diffusion_attn_debug.yaml` uses `src.models.DiffusionAttentionDebugModel`
- the normal model inherits `BaseDiffusionModel`
- the debug model inherits `DebuggableBaseDiffusionModel`

What the base model files do:
- `src/models/base_model.py`: normal training, validation, loss, sampling, checkpointing, and metric entry points
- `src/models/base_model_debuggable.py`: debug-only extensions such as extra shape/metric diagnostics, optional oracle paths, and fixed-noise sampling hooks
- `src/models/base_model_debug.py`: small debug helper functions used only by the debug-capable model
- `src/models/base_model_oracle.py`: oracle-only helper functions used by the debug-capable model
- `src/models/base_model_eval.py`: shared metric-evaluation and visualization helpers used by both normal and debug models


Note:
- if you train the normal model, debug flags in `visual.*` or `model.oracle_cfg.*` will not activate debug-only code paths

`create_dataset.py`:
- root config: `src/configs/create_dataset.yaml`
- config groups used for dataset creation:
- `data=...`: where the generated dataset is written and which `.dvc` file is updated
- `dataset_creation=...`: how the dataset is generated from raw Waymo data for each split

Inside `dataset_creation=...`, each split (`train`, `val`, `test`) controls:
- `raw_data_url`: source TFRecord shard(s)
- `waymax_conf_version`: Waymo/Waymax dataset version
- `num_states`: how many scenes to process
- `max_num_objects`: scene filtering limit before preprocessing
- `extract_scene`: whether to extract scene data
- `preprocessing.*`: preprocessing parameters such as object cap, polyline limits, current index, point count, log transform, and history removal

Rule of thumb:
- change `data=...` when you want a different saved dataset artifact
- change `dataset_creation=...` when you want different dataset contents
- change `model=...`, `trainer.*`, `dataloaders.*`, `metrics`, `logger`, or `visual` when you want different training behavior
