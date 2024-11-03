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
%cd ~
%mkdir {HOMEFOLDER}models/mymodel/
%cd {HOMEFOLDER}models/mymodel/
%pwd

# Download pre-trained model weights
import tarfile
download_tar = 'https://downloads.limelightvision.io/models/' + pretrained_checkpoint
!wget {download_tar}
tar = tarfile.open(pretrained_checkpoint)
tar.extractall()
tar.close()

# Download training configuration file for model
download_config = 'https://downloads.limelightvision.io/models/' + base_pipeline_file
!wget {download_config}
%cd ~

# Set training parameters for the model
num_steps = 40000
checkpoint_every = 2000
batch_size = 16