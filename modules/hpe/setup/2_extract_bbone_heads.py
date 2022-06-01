# Freely inspired by https://github.com/tobibaum/metrabs_trt
# This script saves the BackBone from MetrABS with signature and exports the weight and bias of the heads as numpy

import tensorflow as tf
import numpy as np
import sys
import os

try:
    sys.path.append('modules/hpe/assets/metrabs/src')
    from backbones.efficientnet.effnetv2_model import *
    import backbones.efficientnet.effnetv2_utils as effnet_util
    import tfu
except ImportError as e:
    print("Clone https://github.com/isarandi/metrabs inside modules/hpe/assets first!")
    exit(-1)

if not os.path.exists('modules/hpe/modules/raws/metrabs_eff2l_y4'):
    print("Download weights from https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_y4.zip")
    print("Or from model zoo https://github.com/isarandi/metrabs/blob/master/docs/MODELS.md")
    print("Extract them and put them into modules/hpe/modules/raws")
    exit(-1)

BATCH_SIZE = 5

# Input names
model_name = 'efficientnetv2-l'
model_folder = 'modules/hpe/modules/raws/metrabs_eff2l_y4'

# Output names
bbone_name = "modules/hpe/modules/signatured/bbone{}".format(BATCH_SIZE)
head_name = "modules/hpe/modules/numpy/head"

# Load model
model = tf.saved_model.load(model_folder)
mod_vars = model.crop_model.variables
name_shape = [['/'.join(v.name.split('/')[1:]), v.shape] for v in mod_vars if 'tpu' not in v.name]

####################
# EXTRACT BACKBONE #
####################
effnet_util.set_batchnorm(effnet_util.BatchNormalization)
tfu.set_data_format('NHWC')
tfu.set_dtype(tf.float32)

mod = get_model(model_name, include_top=False, pretrained=False, with_endpoints=False)
mod.set_weights(mod_vars[:-4])


@tf.function()
def my_predict(my_prediction_inputs, **kwargs):
    prediction = mod(my_prediction_inputs, training=False)
    return {"prediction": prediction}


my_signatures = my_predict.get_concrete_function(
    my_prediction_inputs=tf.TensorSpec([BATCH_SIZE, 256, 256, 3], dtype=tf.float32, name="images")
)

tf.saved_model.save(mod, bbone_name, signatures=my_signatures)


#################
# EXTRACT HEADS #
#################
np.save(head_name+'_weights.npy', model.crop_model.heatmap_heads.conv_final.variables.weights[0].numpy())
np.save(head_name+'_bias.npy', model.crop_model.heatmap_heads.conv_final.variables.weights[1].numpy())
