#!/usr/bin/python3

import os
import numpy

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

TRAIN = 'train'
TEST = 'test'
VALID = 'valid'
detection_elements = ['red', 'blue', 'yellow']

train_dataset = object_detector.DataLoader.from_pascal_voc(
	TRAIN,
	TRAIN,
	detection_elements
)

test_dataset = object_detector.DataLoader.from_pascal_voc(
	TEST,
	TEST,
	detection_elements
)

valid_dataset = object_detector.DataLoader.from_pascal_voc(
    VALID,
    VALID,
    detection_elements
)

# Create Model via Model Maker (if model maker does not a good job, we will need to make our own :P )
# spec is set to 'efficient_lite0' in order to optimize performance on SDK when using TFLite Java
spec = model_spec.get('efficientdet_lite0')
model = object_detector.create(
	train_dataset,
	batch_size=4,
	train_whole_model=True,
	epochs=20,
	validation_data=valid_dataset,
    	model_spec=spec
)

# Evaluate the Model and Export it as a  .TFLite file for use on the SDK
model.evaluate(test_dataset)
model.export(export_dir='.', tflite_filename='specimens.tflite')

# End of Model Construction
