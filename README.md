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
To authenticate to google account for data downloading(one time):

1. Apply for [Waymo Open Dataset](https://waymo.com/open) access.
2. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
3. Run ```gcloud auth login <your_email>``` with the same email used for step 1.
4. Run ```gcloud auth application-default login```.

### DVC setup
Processed datasets are tracked with DVC. The `.dvc` files are stored in Git, while the actual `.pkl` files live locally or in the configured DVC remote.

On a new machine:
```bash
uv run bash scripts/setup_dvc_remote.sh myremote <remote-url>
uv run dvc pull
```

If you only need one split:
```bash
uv run dvc pull data/train_processed_v1.pkl.dvc
uv run dvc pull data/val_processed_v1.pkl.dvc
uv run dvc pull data/test_processed_v1.pkl.dvc
```

After that you can run training normally:
```bash
uv run python train.py
```

### Dataset creation
To build processed train/val/test datasets from raw Waymo data:
```bash
uv run python create_dataset.py
```

This script:
- creates the processed `.pkl` files
- runs `dvc add`
- stages the generated `.dvc` files in Git
- runs `dvc push` if a working DVC remote is configured

Main configs:
- dataset artifact paths: `src/configs/data/processed_v1.yaml`
- dataset creation settings: `src/configs/dataset_creation/default.yaml`

You can override settings with Hydra, for example:
```bash
uv run python create_dataset.py dataset_creation.train.num_states=100
uv run python create_dataset.py dataset_creation.val.max_num_objects=16
```

### Data predownload
If you want to download specific path from waymax (Motion)[https://waymo.com/open/download/] dataset run:
```
mkdir -p ./data/training
gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training ./data/training
or
```
```
mkdir -p ./data/training
gsutil cp gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000 ./data/training/

```

### Data visulization
 To run visualization
```
uv run visualise_data.py
```

### Metrics used
####  Overlap (Collision Rate)

Shows fraction of time during which ego vehicle’s bounding box overlaps with any other object.

#### Offroad Rate

Shows fraction of time during which the ego vehicle leaves the drivable road area.

#### Log Divergence (ADE)

Log divergence measures how far the simulated trajectory deviates from the logged (ground-truth) trajectory over time.

#### Collision

A binary metric identifying if collisions happen between agents.


### Example

![Rollout](media/batch_rollout.gif)

#### log_divergence: 60.71818017959595

#### offroad: 0.25

#### overlap: 0.75
