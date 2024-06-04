#!/bin/bash

# Set up venv environment manually
#  each shell needs to have `source venv/bin/activate` run at some time.

python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

