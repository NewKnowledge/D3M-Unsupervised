#!/bin/bash -e 

# clone all files not managed by git LFS locally
git lfs clone https://gitlab.datadrivendiscovery.org/d3m/datasets.git -X "*"
cd datasets

# edit this list of datasets to the datasets you want to download locally and mount to your image
Datasets=('LL1_Adiac' 'LL1_ArrowHead' '66_chlorineConcentration' 'LL1_CinC_ECG_torso' 'LL1_Cricket_Y' 'LL1_ECG200' 'LL1_ElectricDevices' 'LL1_FISH' 'LL1_FaceFour' 'LL1_FordA' 'LL1_HandOutlines' 'LL1_Haptics' 'LL1_ItalyPowerDemand' 'LL1_Meat' 'LL1_OSULeaf')

# download files mananged by git LFS for each of desired datasets
for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done

# pull latest program image
sudo docker pull registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7 

# run program image
    # --rm
        # automatically remove the container when it exits
    # -t -i
        # allocate a tty for the container process, which allows you to access interactive process (like a shell)
    # --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/datasets,target=/datasets
        # mount your local datasets and all downloaded datasets therein to the image
    # bash
        # bash into running container
sudo docker run --rm -t -i --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/datasets,target=/datasets registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7 bash