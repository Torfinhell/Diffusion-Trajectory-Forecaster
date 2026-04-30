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
```

How it works:
- the repository is mounted into the container at `/app`
- your code, checkpoints, outputs, and local changes stay on the host machine
- the container uses its own virtual environment at `/opt/venv`, so Docker does not recreate or modify your host `.venv`
- the helper script runs the container with your host UID/GID so generated files remain writable by your user and Git can stage them
- container-side cache and auth files are stored in gitignored `.docker-cache/`


Notes:
- rebuild the image after this change so the container environment is created under `/opt/venv`

### To authenticate to google account for data downloading(one time):

1. Apply for [Waymo Open Dataset](https://waymo.com/open) access.
2. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
3. Run ```gcloud auth login <your_email>``` with the same email used for step 1.
4. Run ```gcloud auth application-default login```.

### Dataset creation
To build processed train/val/test datasets from raw Waymo data:
```bash
 uv run python -m scripts.create_dataset -cn=small_no_scenes
```

### DVC setup
Processed datasets are tracked with DVC as directory artifacts. Git stores the `.dvc` metadata files, while the actual `.wds` files live locally or in the configured DVC remote.

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

To track new dataset run:
``` bash
uv run scripts/add_local_dataset_to_dvc.sh path_to_dataset_folder
```
it adds dataset to dvc, push it to remote storage and stages .dvc file.

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

### Train from S3-hosted WebDataset shards:
```bash
uv run python train.py \
  feat_extract=small_no_scenes_s3
```
To use online training AWS cli is required:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```
To provide credentials run
``` bash
aws configure
```
It asks next values:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Remote WebDataset notes:
- if `processed_path` starts with `s3://`, the dataset is treated as remote
- during creation, shards are written to `local_cache_path`, uploaded to S3, and removed locally
- during training, shards are streamed from S3 with `aws s3 cp`
