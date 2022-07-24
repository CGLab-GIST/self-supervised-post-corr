
#  Copyright (c) 2022 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import image_io.exr as exr
import numpy as np
import tensorflow as tf
import argparse

from tensorflow.keras import Model, layers
from models.post_correction import post_corrector_sampler, post_corrector_configure, post_corrector_assign_buffers
from models.post_correction import post_corrector_color_buffers_allocate, post_corrector_color_buffers_deallocate
from models.post_correction import post_corrector_gbuffers_allocate, post_corrector_gbuffers_deallocate
from models.post_correction import post_corrector_revised_kernel

# =============================================================================
# Arguments for running different kinds of images
#  - Example running code
#    python main.py --scene dragon --spp 128 --deno afgsa

parser = argparse.ArgumentParser(description='Arguments for Self-Supervised Post-Correction')
parser.add_argument('--scene', help='Scene name', type=str, required=True)
parser.add_argument('--spp', help='Samples per pixel', type=str, required=True)
parser.add_argument('--deno', help='Denoiser', type=str, required=True)
parser.add_argument('--epoch', help='Epoch', default=20, type=int)
parser.add_argument('--print', help='Print training progress (0 or 1)', default=0, type=int)
parser.add_argument('--inputdir', help='Directory for input data', default='../data/__example_scenes__/', type=str)
parser.add_argument('--outputdir', help='Directory for output data', default='../data/', type=str)
args = vars(parser.parse_args())


# =============================================================================
# Other functionality
def getRelL2(inputs, ref):
    num = tf.square(tf.subtract(inputs, ref))
    denom = tf.reduce_mean(ref, axis=3, keepdims=True)
    relL2 = num / (denom * denom + 1e-2)
    return tf.reduce_mean(relL2)


# =============================================================================
# Configuration setting
CURR_SCENE_NAME     = args['scene']
CURR_SPP            = args['spp'] + 'spp'
CURR_DENO_IMG       = args['deno']
PRINT_ON            = args['print']

COLOR_DIM           = 3
NUM_AUX_FEATS       = 3
AUX_FEATS_DIM       = 7
INPUT_FEATS_DIM     = COLOR_DIM * 2 + AUX_FEATS_DIM
KERNEL_SIZE         = 19
KERNEL_SIZE_SQR     = KERNEL_SIZE * KERNEL_SIZE
NUM_EPOCH           = args['epoch']
PATCH_WIDTH         = 128
PATCH_HEIGHT        = 128
STRIDE_WIDTH        = 64
STRIDE_HEIGHT       = 64
BATCH_SIZE          = 16
CONV_KERNEL_SIZE    = 3

INPUT_SCENE_PATH    = os.path.join(args['inputdir'], "input", CURR_DENO_IMG, CURR_SCENE_NAME)
REF_SCENE_PATH      = os.path.join(args['inputdir'], "target", CURR_SCENE_NAME)
OUTPUT_SCENE_PATH   = os.path.join(args['outputdir'], '__' + CURR_DENO_IMG + '__')
SCENE_PREFIX        = CURR_SCENE_NAME + '_' + CURR_SPP

NETOUT_SLOPE_NUM    = 3
NETOUT_SLOPE_DIM    = NETOUT_SLOPE_NUM * COLOR_DIM      # Scale parameters (\beta in Eq. 10)
NETOUT_WGT_DIM      = NUM_AUX_FEATS + 2                 # Bandwidth parameters (\gamma in Eq. 11)
NETOUT_ALPHA_DIM    = 1                                 # Center weight (\tau in Eq. 11)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not os.path.exists(OUTPUT_SCENE_PATH):
    os.makedirs(OUTPUT_SCENE_PATH)


# Buffer reading for post-correction
inputDir = os.path.join(INPUT_SCENE_PATH, SCENE_PREFIX)
randImgA = np.maximum(np.array([exr.read(inputDir + '_randImgA.exr')]), 0.0)
randImgB = np.maximum(np.array([exr.read(inputDir + '_randImgB.exr')]), 0.0)
# Noisy image with full buffers reading for comparison with ours
randImg = (randImgA + randImgB) / 2.0

denoImgA = np.maximum(np.array([exr.read(inputDir + '_denoImgA.exr')]), 0.0)
denoImgB = np.maximum(np.array([exr.read(inputDir + '_denoImgB.exr')]), 0.0)
# Denoised image with full buffers reading for comparison with ours
denoImg = np.maximum(np.array([exr.read(inputDir + '_denoImg.exr')]), 0.0)

albedoA = np.array([exr.read(inputDir + '_albedoA.exr')])
albedoB = np.array([exr.read(inputDir + '_albedoB.exr')])
albedo = np.array([exr.read(inputDir + '_albedo.exr')])
normalA = np.array([exr.read(inputDir + '_normalA.exr')])
normalB = np.array([exr.read(inputDir + '_normalB.exr')])
normal = np.array([exr.read(inputDir + '_normal.exr')])
visibilityA = np.array([exr.read(inputDir + '_visibilityA.exr')])[:,:,:,0:1]
visibilityB = np.array([exr.read(inputDir + '_visibilityB.exr')])[:,:,:,0:1]
visibility = np.array([exr.read(inputDir + '_visibility.exr')])[:,:,:,0:1]

albedoA = np.concatenate((albedoA, visibilityA), axis=-1)
albedoB = np.concatenate((albedoB, visibilityB), axis=-1)
albedo = np.concatenate((albedo, visibility), axis=-1)

# Reference reading for checking numerical accuracy
refDir = os.path.join(REF_SCENE_PATH, CURR_SCENE_NAME + '_ref.exr')
refImg = np.maximum(np.array([exr.read(refDir)]), 0.0)

# Changing the learning rate according to the estimated variance of noisy estimates
meanRandVar = np.mean(np.square(randImgA - randImgB) / 4.0)
LEARNING_RATE = 0.01 * np.sqrt(meanRandVar)

print('%s scene, %s, %d epoch, %s, lr %f' % (CURR_SCENE_NAME, CURR_SPP, NUM_EPOCH, CURR_DENO_IMG, LEARNING_RATE))


# =============================================================================
# Network architecture
class SelfSupervisedPostCorrect:
    def __init__(self):
        inputs = layers.Input(shape=(None, None, INPUT_FEATS_DIM))

        c1 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(inputs)
        c2 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(c1)
        c3 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(c2)
        c4 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(c3)
        c5 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(c4)
        c6 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(c5)
        c7 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(c6)
        c8 = layers.Conv2D(filters=16, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same')(c7)
        params = layers.Conv2D(filters=NETOUT_SLOPE_DIM + NETOUT_WGT_DIM + NETOUT_ALPHA_DIM, kernel_size=CONV_KERNEL_SIZE, padding='same')(c8)

        slopes, bands, alpha = tf.split(params, [NETOUT_SLOPE_DIM, NETOUT_WGT_DIM, NETOUT_ALPHA_DIM], axis=-1)
        bands = tf.math.reciprocal(tf.math.square(tf.nn.softplus(bands)) + 0.0001)    
        alpha = tf.nn.softplus(alpha) + 0.0001
        slopes = tf.nn.tanh(slopes)

        self.model = Model(inputs=inputs, outputs=[slopes, bands, alpha])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.model.compile()
        self.model.summary()


    @tf.function
    def train_step(self, _inputImgsA, _inputImgsB, _randA, _randB, _tgtA, _tgtB, _refImg, _sampledPatchIdx):
        with tf.GradientTape() as g:
            # Post-correction function w/ cross-bilateral weighting (Eqs. 10 and 11 in paper)
            _slopeA, _wgtA, _alphaA = self.model(_inputImgsA)
            _outImgA = post_corrector_revised_kernel(_sampledPatchIdx, _slopeA, _wgtA, _alphaA, buf_idx=1, training=True)
            _slopeB, _wgtB, _alphaB = self.model(_inputImgsB)
            _outImgB = post_corrector_revised_kernel(_sampledPatchIdx, _slopeB, _wgtB, _alphaB, buf_idx=2, training=True)

            # Self-supervised loss function (Eqs. 8 and 9 in paper)
            lossDenA = tf.square(tf.reduce_mean(_tgtB, axis=-1, keepdims=True)) + 1e-2
            lossDenB = tf.square(tf.reduce_mean(_tgtA, axis=-1, keepdims=True)) + 1e-2
            lossA = (tf.square(_outImgA - _randB)) / (lossDenA)
            lossB = (tf.square(_outImgB - _randA)) / (lossDenB)
            loss = 0.5 * (lossA + lossB)
            avgLoss = tf.reduce_mean(loss)
        dL_dk = g.gradient(avgLoss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(dL_dk, self.model.trainable_variables))

        # For the purpose of checking numerical error 
        # (Note that the reference should not be used for our training step)
        testLoss = tf.constant(0.0)
        if PRINT_ON:
            _outImg = 0.5 * (_outImgA + _outImgB)
            refImgOneCh = tf.reduce_mean(_refImg, axis=3, keepdims=True)
            testLoss = tf.reduce_mean(tf.square(_outImg - _refImg) / (tf.square(refImgOneCh) + 1e-2))    

        return avgLoss, testLoss


    @tf.function
    def test_step(self, _inputImgs, bufIdx, _sampledPatchIdx):
        _slope, _wgt, _alpha = self.model(_inputImgs)
        _outImg = post_corrector_revised_kernel(_sampledPatchIdx, _slope, _wgt, _alpha, buf_idx=bufIdx, training=False)
        return _outImg


# =============================================================================
# The functions of on the fly training and test code
postCorrector = SelfSupervisedPostCorrect()

def main_train_function(_batchSize, _sampledPatchIdx):
    # Image patch extraction
    x_randPatchA = post_corrector_sampler(_sampledPatchIdx, img_type='rand', buf_idx=1, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=3)
    x_denoPatchA = post_corrector_sampler(_sampledPatchIdx, img_type="deno", buf_idx=1, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=3)
    x_albedoPatchA = post_corrector_sampler(_sampledPatchIdx, img_type='albedo', buf_idx=1, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=4)
    x_normalPatchA = post_corrector_sampler(_sampledPatchIdx, img_type='normal', buf_idx=1, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=3)

    x_randPatchB = post_corrector_sampler(_sampledPatchIdx, img_type='rand', buf_idx=2, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=3)
    x_denoPatchB = post_corrector_sampler(_sampledPatchIdx, img_type="deno", buf_idx=2, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=3)
    x_albedoPatchB = post_corrector_sampler(_sampledPatchIdx, img_type='albedo', buf_idx=2, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=4)
    x_normalPatchB = post_corrector_sampler(_sampledPatchIdx, img_type='normal', buf_idx=2, \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=3)

    y_refPatch = post_corrector_sampler(_sampledPatchIdx, img_type="ref", \
        batch_size=_batchSize, patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, dim_size=3)

    # Setting network inputs
    x_netInputA = tf.concat([x_randPatchA, x_albedoPatchA, x_normalPatchA, x_denoPatchA], axis=-1)
    x_netInputB = tf.concat([x_randPatchB, x_albedoPatchB, x_normalPatchB, x_denoPatchB], axis=-1)
        
    loss, testLoss = postCorrector.train_step(x_netInputA, x_netInputB, x_randPatchA, x_randPatchB, \
        x_denoPatchA, x_denoPatchB, y_refPatch, _sampledPatchIdx)

    return loss, testLoss


def main_test_function():
    sampledPatchIdx = tf.range(0, 1, 1)
    netInputA = tf.concat([randImgA, albedoA, normalA, denoImgA], axis=-1)
    netInputB = tf.concat([randImgB, albedoB, normalB, denoImgB], axis=-1)
    finalOutImgA = postCorrector.test_step(netInputA, 1, sampledPatchIdx)
    finalOutImgB = postCorrector.test_step(netInputB, 2, sampledPatchIdx)
    finalOutImg = tf.math.maximum(0.5 * (finalOutImgA + finalOutImgB), 0.0)
    return finalOutImg


# =============================================================================
# Main training part
imgShape = tf.shape(randImg)
nPatchY = int(tf.math.ceil(imgShape[1] / STRIDE_HEIGHT))
nPatchX = int(tf.math.ceil(imgShape[2] / STRIDE_WIDTH))
numTotalPatches = nPatchX * nPatchY
print('Number of patches            : %d' % (numTotalPatches))
if numTotalPatches % BATCH_SIZE != 0:
    print("Warning! The number of patches are not the multiples of the batch size, and it may increase the running time")

post_corrector_configure(img_height=imgShape[1], img_width=imgShape[2], 
                   patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH, 
                   stride_height=STRIDE_HEIGHT, stride_width=STRIDE_WIDTH, 
                   batch_size=BATCH_SIZE, win_size=KERNEL_SIZE)
post_corrector_color_buffers_allocate(randImg, denoImg, refImg)
post_corrector_gbuffers_allocate(albedo, normal)
post_corrector_assign_buffers(img_a=randImgA, img_b=randImgB, img_type='rand')
post_corrector_assign_buffers(img_a=denoImgA, img_b=denoImgB, img_type='deno')
post_corrector_assign_buffers(img_a=albedoA, img_b=albedoB, img_type='albedo')
post_corrector_assign_buffers(img_a=normalA, img_b=normalB, img_type='normal')

# Initial running
randArr = tf.random.uniform([BATCH_SIZE], minval=0, maxval=numTotalPatches, dtype=tf.dtypes.int32)
_, _ = main_train_function(BATCH_SIZE, randArr)
_ = main_test_function()

# Starting main training phase from here
startTime = time.time()
tmpArr = tf.range(0, numTotalPatches, 1)
numStepPerEpoch = int(tf.math.ceil(numTotalPatches / BATCH_SIZE))

for epoch in range(NUM_EPOCH):    
    accLoss = 0.0
    accTestLoss = 0.0
    tmpArr = tf.random.shuffle(tmpArr)
    epochStartTime = time.time()
    for batchIdx in range(numStepPerEpoch):
        sampledPatchIdx = tmpArr[batchIdx*BATCH_SIZE : tf.math.minimum((batchIdx+1)*BATCH_SIZE, tf.size(tmpArr))]
        batchSize = tf.size(sampledPatchIdx)
        loss, testLoss = main_train_function(batchSize, sampledPatchIdx)
        accLoss += loss 
        accTestLoss += testLoss
    epochEndTime = time.time() - epochStartTime
    if PRINT_ON:
        print('epoch %d (%.3fsec), avgTrainLoss %.6f, avgTestLoss %.6f...' % (epoch, epochEndTime, accLoss / numStepPerEpoch, accTestLoss / numStepPerEpoch))    

trainRunningTime = time.time() - startTime
print("Training time (%d epoch)     : %.3f sec" % (NUM_EPOCH, trainRunningTime))


# =============================================================================
# Final inference
startTime = time.time()
finalOutImg = main_test_function()
inferenceTime = time.time() - startTime
print("Inference time               : %.3f sec" % (inferenceTime))
print("Total running time           : %.3f sec" % (trainRunningTime + inferenceTime))


# =============================================================================
# Saving outputs
relL2RandImg = getRelL2(randImg, refImg)
relL2DenoImg = getRelL2(denoImg, refImg)
relL2OutImg = getRelL2(finalOutImg, refImg)

outputDir = os.path.join(OUTPUT_SCENE_PATH, SCENE_PREFIX)
exr.write(outputDir + '_in_randImg_' + '{:.6f}'.format(relL2RandImg) + '.exr', randImg[0])
exr.write(outputDir + '_in_denoImg_' + '{:.6f}'.format(relL2DenoImg) + '.exr', denoImg[0])
exr.write(outputDir + '_output_' + '{:.6f}'.format(relL2OutImg) + '.exr', finalOutImg[0].numpy())
exr.write(outputDir + '_ref.exr', refImg[0])

print("Relative L2 of noisyImg      : %.6f" % (relL2RandImg))
print("Relative L2 of denoisedImg   : %.6f" % (relL2DenoImg))
print("Relative L2 of ours          : %.6f" % (relL2OutImg))

post_corrector_color_buffers_deallocate()
post_corrector_gbuffers_deallocate()
print('Done!')