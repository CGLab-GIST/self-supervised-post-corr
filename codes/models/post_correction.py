
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


import numpy as np
import tensorflow as tf

_module = tf.load_op_library('./models/post_correction_lib.so')

# ===========================================================
#    Functionalities for self-supervised post-correction
# ===========================================================
post_corrector_sampler = _module.post_corrector_sampler
post_corrector_configure = _module.post_corrector_configure
post_corrector_assign_buffers = _module.post_corrector_assign_buffers
post_corrector_color_buffers_allocate = _module.post_corrector_color_buffers_allocate
post_corrector_color_buffers_deallocate = _module.post_corrector_color_buffers_deallocate
post_corrector_gbuffers_allocate = _module.post_corrector_gbuffers_allocate
post_corrector_gbuffers_deallocate = _module.post_corrector_gbuffers_deallocate
# ===========================================================


# ===========================================================
#    Post-correction function w/ cross-bilateral weighting
#        (Eq. 10 in paper)           (Eq. 11 in paper)
# ===========================================================
@tf.RegisterGradient("PostCorrectorRevisedKernel")
def _post_corrector_revised_kernel_grad(op, grad):
    idxList = op.inputs[0]
    slope = op.inputs[1]
    wgt = op.inputs[2]
    alpha = op.inputs[3]
    bufIdx = op.get_attr('buf_idx')
    training = op.get_attr('training')  
    grad_slope, grad_wgt, grad_alpha = _module.post_corrector_revised_kernel_grad(grad, idxList, slope, wgt, alpha, buf_idx=bufIdx, training=training)
    return [None, grad_slope, grad_wgt, grad_alpha]

post_corrector_revised_kernel = _module.post_corrector_revised_kernel
# ===========================================================