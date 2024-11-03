#!/bin/bash
mamba create -n limelight 'python>=3.9,<3.10'
mamba activate limelight
mamba init
. ~/.bashrc
mamba activate limelight
pip install tflite_model_maker tensorflow
apt update
apt install libusb-1.0-0-dev
python construct.py
