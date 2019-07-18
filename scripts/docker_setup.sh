#!/bin/bash -e

cd /
# change then line below to your branch: https://gitlab.com/<gitlab_username>/primitives
git clone https://gitlab.com/jgleason/primitives
cd primitives
git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
git pull upstream master
