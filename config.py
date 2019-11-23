import os
import math


PROJECT_DIR = '.'
DATA_DIR = os.path.join('./', 'data')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
SNAPSHOT_DIR = os.path.join(MODEL_DIR, 'snapshots')

AFFECTNET_RAW_DIR = './datasets/AffectNet'
AFFECTNET_ROOT = './datasets/AffectNet'

VGGFACE2_ROOT = './datasets/VGGFace2'
VGGFACE2_ROOT_LOCAL = './datasets/VGGFace2'

W300_ROOT = './datasets/300W'
W300_ROOT_LOCAL = './datasets/W300'

AFLW_ROOT = './datasets/AFLW/aflw'
AFLW_ROOT_LOCAL = './datasets/AFLW'

WFLW_ROOT = './datasets/WFLW'
WFLW_ROOT_LOCAL = './datasets/WFLW'

ARCH = 'resnet'

INPUT_SCALE_FACTOR = 2
INPUT_SIZE = 128 * INPUT_SCALE_FACTOR
CROP_SIZE = math.ceil(INPUT_SIZE * 2**0.5)  # crop size equals input diagonal, so images can be fully rotated
CROP_BORDER = (CROP_SIZE - INPUT_SIZE) // 2

CROP_BY_EYE_MOUTH_DIST = False
CROP_ALIGN_ROTATION = False
CROP_SQUARE = True
# crop resizing params for cropping based on landmarks
CROP_MOVE_TOP_FACTOR = 0.2       # move top by factor of face heigth in respect to eye brows
CROP_MOVE_BOTTOM_FACTOR = 0.12   # move bottom by factor of face heigth in respect to chin bottom point

MIN_OPENFACE_CONFIDENCE = 0.4

ENCODER_LAYER_NORMALIZATION = 'batch'
DECODER_LAYER_NORMALIZATION = 'batch'
ENCODING_DISTRIBUTION = 'normal'

DECODER_FIXED_ARCH = True
DECODER_PLANES_PER_BLOCK = 1

# Autoencoder losses
TRAIN_ENCODER = True
TRAIN_DECODER = True

RGAN = False
UPDATE_DISCRIMINATOR_FREQ = 4
UPDATE_ENCODER_FREQ = 1

WITH_ZGAN = True
WITH_GAN = True
WITH_GEN_LOSS = True
WITH_LANDMARK_LOSS = False
WITH_SSIM_LOSS = True

WITH_HIST_NORM = False

# Recontruction loss
W_RECON = 1.0
W_SSIM = 60.0

EMBEDDING_DIMS = 99

WITH_FACE_MASK = False
WITH_RECON_ERROR_WEIGHTING = False

WEIGHT_RECON_LOSS = 2.0

