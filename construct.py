#!/usr/bin/python3

# ISSUE: No validation or testing datasets are made in the pascal_VOC format

# Packages
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

# Global Constants
path_to_train = 'train'
path_to_test = 'test'
path_to_valid = 'valid'
detection_elements = ['specimen_block_red','specimen_block_blue','specimen_block_yellow']

# Declare Training Dataset (what the model will train upon)
train_dataset = object_detector.DataLoader.from_pascal_voc(
	path_to_train,
	path_to_train,
	detection_elements
)

# Declare Testing Dataset (what the model will use to predict)
test_dataset = object_detector.DataLoader.from_pascal_voc(
	path_to_test,
	path_to_test,
	detection_elements
)

val_dataset = object_detector.DataLoader.from_pascal_voc(
	path_to_valid,
	path_to_valid,
	detection_elements
)

# Create Model via Model Maker (if model maker does not a good job, we will need to make our own :P )
# spec is set to 'efficient_lite0' in order to optimize performance on SDK when using TFLite Java
spec = model_spec.get('efficientdet_lite0')
model = object_detector.create(
	train_dataset,
	batch_size=1,
	train_whole_model=True,
	epochs=20,
	validation_data=val_dataset,
    model_spec=spec
)

print(f"Train classes: {train_dataset.label_map}")
print(f"Test classes: {test_dataset.label_map}")
print(f"Validation classes: {val_dataset.label_map}")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Evaluate the Model and Export it as a  .TFLite file for use on the SDK
model.evaluate(test_dataset)
model.export(export_dir='.', tflite_filename='specimens.tflite')
# End of Model Construction