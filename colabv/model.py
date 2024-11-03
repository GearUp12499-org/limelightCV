# limelight_detector_training.py

import os
import re
import shutil
import tarfile
import subprocess
import tensorflow as tf
from google.colab import drive

# Configurations
MLENVIRONMENT = "COLAB"
HOMEFOLDER = "/content/" if MLENVIRONMENT == "COLAB" else "./"
MODEL_DIR = os.path.join(HOMEFOLDER, "training_progress/")
FINALOUTPUTFOLDER = os.path.join(HOMEFOLDER, "final_output/")
CHECKPOINT_DIR = os.path.join(HOMEFOLDER, "training_progress/")
PIPELINE_FILE = 'pipeline_file.config'
BATCH_SIZE = 16
NUM_STEPS = 40000
CHECKPOINT_EVERY = 2000
CHOSEN_MODEL = 'ssd-mobilenet-v2'

# Model details
MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
}

# Model paths
model_name = MODELS_CONFIG[CHOSEN_MODEL]['model_name']
pretrained_checkpoint = MODELS_CONFIG[CHOSEN_MODEL]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[CHOSEN_MODEL]['base_pipeline_file']

# Environment setup
def setup_environment():
    if MLENVIRONMENT == "COLAB":
        os.environ["HOMEFOLDER"] = "/content/"
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/tensorflow/models"])
        subprocess.run(["git", "fetch", "--depth", "1", "origin", "ad1f7b56943998864db8f5db0706950e93bb7d81"], cwd="models")
        subprocess.run(["git", "checkout", "ad1f7b56943998864db8f5db0706950e93bb7d81"], cwd="models")
        subprocess.run(["pip", "install", os.path.join(HOMEFOLDER, "models/research/")])
        subprocess.run(["pip", "install", "tensorflow==2.15.0"])
        drive.mount('/content/gdrive')

# Dataset handling
def get_dataset_path():
    dataset_path = '/content/your_dataset_path.tfrecord.zip'
    subprocess.run(["unzip", dataset_path, "-d", "/content/"])

# Helper to configure tfrecord paths
def find_files(directory, pattern):
    matches = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if re.match(pattern, filename):
                matches.append(os.path.join(root, filename))
    return matches

# Training configurations
def configure_pipeline():
    train_record_fname, val_record_fname, label_map_pbtxt_fname = '/content/train.record', '/content/val.record', '/content/label_map.pbtxt'
    fine_tune_checkpoint = f"{HOMEFOLDER}models/mymodel/{model_name}/checkpoint/ckpt-0"
    with open(base_pipeline_file) as f:
        s = f.read()
    with open(PIPELINE_FILE, 'w') as f:
        s = re.sub('fine_tune_checkpoint: ".*?"', f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)
        s = re.sub('input_path: ".*?train.*?"', f'input_path: "{train_record_fname}"', s)
        s = re.sub('input_path: ".*?val.*?"', f'input_path: "{val_record_fname}"', s)
        s = re.sub('label_map_path: ".*?"', f'label_map_path: "{label_map_pbtxt_fname}"', s)
        s = re.sub('batch_size: [0-9]+', f'batch_size: {BATCH_SIZE}', s)
        s = re.sub('num_steps: [0-9]+', f'num_steps: {NUM_STEPS}', s)
        f.write(s)

# Model Training
def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    subprocess.run([
        "python", f"{HOMEFOLDER}models/research/object_detection/model_main_tf2.py",
        "--pipeline_config_path", PIPELINE_FILE,
        "--model_dir", MODEL_DIR,
        "--alsologtostderr",
        "--checkpoint_every_n", str(CHECKPOINT_EVERY),
        "--num_train_steps", str(NUM_STEPS),
        "--sample_1_of_n_eval_examples=1"
    ])

# Convert Model to TFLite
def export_model():
    if os.path.exists(FINALOUTPUTFOLDER):
        shutil.rmtree(FINALOUTPUTFOLDER)
    os.makedirs(FINALOUTPUTFOLDER, exist_ok=True)
    subprocess.run([
        "python", f"{HOMEFOLDER}models/research/object_detection/export_tflite_graph_tf2.py",
        "--trained_checkpoint_dir", MODEL_DIR,
        "--output_directory", FINALOUTPUTFOLDER,
        "--pipeline_config_path", PIPELINE_FILE
    ])

    converter = tf.lite.TFLiteConverter.from_saved_model(FINALOUTPUTFOLDER + '/saved_model')
    tflite_model = converter.convert()
    model_path_32bit = os.path.join(FINALOUTPUTFOLDER, 'limelight_neural_detector_32bit.tflite')
    with open(model_path_32bit, 'wb') as f:
        f.write(tflite_model)

# Run all steps
if __name__ == "__main__":
    setup_environment()
    get_dataset_path()
    configure_pipeline()
    train_model()
    export_model()
    print(f"Model training and export complete. Check output at {FINALOUTPUTFOLDER}")