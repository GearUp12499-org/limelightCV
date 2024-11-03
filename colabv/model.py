import tarfile
import os
import shutil

tmpModelPath ='/content/models'
if os.path.exists(tmpModelPath) and os.path.isdir(tmpModelPath):
  shutil.rmtree(tmpModelPath)

MLENVIRONMENT="COLAB"
os.system("git clone --depth 1 https://github.com/tensorflow/models")
os.system("cd models && git fetch --depth 1 origin ad1f7b56943998864db8f5db0706950e93bb7d81 && git checkout ad1f7b56943998864db8f5db0706950e93bb7d81")

chosen_model = 'ssd-mobilenet-v2'
MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
}
model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

# Create "mymodel" folder for pre-trained weights and configuration files
os.system("cd ~")
os.system("mkdir ~/models/mymodel/")
os.system("cd ~/models/mymodel/")
os.getcwd()

# Download pre-trained model weights
download_tar = 'https://downloads.limelightvision.io/models/' + pretrained_checkpoint
os.system("wget" + download_tar)
tar = tarfile.open(pretrained_checkpoint)
tar.extractall()
tar.close()

# Download training configuration file for model
download_config = 'https://downloads.limelightvision.io/models/' + base_pipeline_file
os.system("wget" + download_config)
os.system("cd ~")

# Set training parameters for the model
num_steps = 40000
checkpoint_every = 2000
batch_size = 16

# Set file locations and get number of classes for config file
pipeline_fname = HOMEFOLDER+'models/mymodel/' + base_pipeline_file
fine_tune_checkpoint = HOMEFOLDER+'models/mymodel/' + model_name + '/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

def get_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    class_names = [category['name'] for category in category_index.values()]
    return class_names

def create_label_file(filename, labels):
    with open(filename, 'w') as file:
        for label in labels:
            file.write(label + '\n')


num_classes = get_num_classes(label_map_pbtxt_fname)
classes = get_classes(label_map_pbtxt_fname)

print('Total classes:', num_classes)
print(classes)


#Generate labels file
create_label_file(HOMEFOLDER + "limelight_neural_detector_labels.txt", classes)