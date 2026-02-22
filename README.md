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
```
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.gpg
gcloud auth login
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
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
