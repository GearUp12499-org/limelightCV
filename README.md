# 12499 Gear Up 2024-25 Limelight CV Repository

## Dependencies

```
(pip) pycocotools
(pip) numpy
(pip) tflite_model_maker
(pip) tensorflow
(system) libusb-1.0-0-dev
```

## Details

Image Format: `Pascal VOC`

Dataset Source: `Roboflow`

Reccomended Image: `https://hub.docker.com/r/condaforge/miniforge3` 

## Setup

```
$ mamba create -n limelight 'python>=3.9,<3.10'
$ mamba activate limelight
$ mamba init
$ source ~/.bashrc
$ mamba activate limelight
$ pip install tflite_model_maker tensorflow
$ apt update
$ apt install libusb-1.0-0-dev
$ python construct.py
```
