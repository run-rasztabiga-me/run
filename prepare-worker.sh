#!/usr/bin/env bash

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "OPENAI_API_KEY is not set"
  exit 1
fi

# upgrade packages
sudo apt update
sudo apt dist-upgrade -y

# install microk8s
sudo snap install microk8s --classic
microk8s status --wait-ready

# enable addons
microk8s enable registry
microk8s enable hostpath-storage
microk8s enable ingress

# install docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh


# clone worker repo
git clone https://github.com/BartlomiejRasztabiga/run.git
cd run
echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
echo "REGISTRY_URL=localhost:32000" >> .env

# install python
sudo apt install python3 python3-pip python3-venv -y

# setup worker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# rootless docker
sudo usermod -aG docker $USER
newgrp docker
