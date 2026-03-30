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
- keep secrets such as `gdrive_client_id` and `gdrive_client_secret` in `.dvc/config.local`
- do not commit `.dvc/config.local`

Typical local setup:
```bash
uv run dvc remote list
uv run dvc remote modify --local myremote gdrive_client_id <YOUR_CLIENT_ID>
uv run dvc remote modify --local myremote gdrive_client_secret <YOUR_CLIENT_SECRET>
```

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

Recommended workflow with Docker:
- run `dvc pull` and `dvc push` on the host, not inside Docker
- run training and evaluation inside Docker
- this avoids repeating the Google Drive browser login flow inside the container

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
