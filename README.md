1. Run updates 

sudo apt-get update -y -qq 
sudo apt-get upgrade -y -qq 
sudo apt-get install -y -qq golang neofetch zsh byobu 
sudo apt-get install -y -qq software-properties-common 
sudo add-apt-repository -y ppa:deadsnakes/ppa 
sudo apt-get install -y -qq python3.12-full python3.12-dev 
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended 
sudo chsh $USER -s /usr/bin/zsh

2. Config the venv 

python3.12 -m venv ~/venv
. ~/venv/bin/activate
pip install -U pip
pip install -U wheel
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U equinox
pip install -U optax
pip install -U grain
pip install -U tqdm
pip install -U typing 
pip install -U tensorflow_datasets
pip install -U PyYAML
pip install tensorflow tensorboard-plugin-profile # For profiling, if you dont want tensorboard profiling for jax, just skip

3. Mount the dataset

login : gcloud auth application-default login

install google fuse 

    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install fuse gcsfuse
    gcsfuse -v

mkdir "$HOME/dataset"
gcsfuse --implicit-dirs s2glc_array_records "$HOME/dataset"

4. Mount the logs dir 