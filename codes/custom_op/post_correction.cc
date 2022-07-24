
//  Copyright (c) 2022 CGLab, GIST. All rights reserved.
//  
//  Redistribution and use in source and binary forms, with or without modification, 
//  are permitted provided that the following conditions are met:
//  
//  - Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation 
//    and/or other materials provided with the distribution.
//  - Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.
//  
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// ===========================================================
//    Functionalities for self-supervised post-correction
// ===========================================================
REGISTER_OP("PostCorrectorSampler")    
  .Input("idx_list: int32")
  .Output("out: float32")
  .Attr("img_type: {'rand', 'deno', 'albedo', 'normal', 'ref'}") 
  .Attr("buf_idx: int = 0")
  .Attr("batch_size: int >= 1")  
  .Attr("patch_height: int >= 1")  
  .Attr("patch_width: int >= 1")    
  .Attr("dim_size: int >= 1")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {    	
	int batchSize, patchHeight, patchWidth, strideSize, dimSize;
	TF_RETURN_IF_ERROR(c->GetAttr("batch_size", &batchSize));
	TF_RETURN_IF_ERROR(c->GetAttr("patch_height", &patchHeight));
	TF_RETURN_IF_ERROR(c->GetAttr("patch_width", &patchWidth));
	TF_RETURN_IF_ERROR(c->GetAttr("dim_size", &dimSize));
		  
    std::vector<shape_inference::DimensionHandle> dims;
    dims.emplace_back(c->MakeDim(batchSize));
    dims.emplace_back(c->MakeDim(patchHeight));
    dims.emplace_back(c->MakeDim(patchWidth));
    dims.emplace_back(c->MakeDim(dimSize));
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  });

REGISTER_OP("PostCorrectorAssignBuffers")  
  .Input("img_a: float32")
  .Input("img_b: float32")
  .Attr("img_type: {'rand', 'deno', 'albedo', 'normal'}")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	shape_inference::ShapeHandle unused;	
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &unused));
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &unused));
	return Status::OK();
  });  
  
REGISTER_OP("PostCorrectorConfigure")  
  .Attr("img_height: int >= 1")  
  .Attr("img_width: int >= 1")    
  .Attr("batch_size: int >= 1")	
  .Attr("patch_height: int >= 1")  
  .Attr("patch_width: int >= 1")
  .Attr("stride_height: int >= 1")  
  .Attr("stride_width: int >= 1")
  .Attr("win_size: int >= 1")   
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	return Status::OK();
  });

REGISTER_OP("PostCorrectorColorBuffersAllocate")  
  .Input("rand: float32")
  .Input("deno: float32")
  .Input("ref: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	shape_inference::ShapeHandle unused;	
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &unused));
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &unused));	
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &unused));
	return Status::OK();
  });

REGISTER_OP("PostCorrectorColorBuffersDeallocate")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	return Status::OK();
  });

REGISTER_OP("PostCorrectorGbuffersAllocate")  
  .Input("albedo: float32")
  .Input("normal: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	shape_inference::ShapeHandle unused;	
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &unused));
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &unused));	
	return Status::OK();
  });

REGISTER_OP("PostCorrectorGbuffersDeallocate")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	return Status::OK();
  });

void PostCorrectorConfigureFunction(int imgHeight, int imgWidth, int patchHeight, int patchWidth, int strideHeight, int strideWidth, int batchSize, int winSize);
void PostCorrectorAssignBuffersFunction(const GPUDevice &_dev, const float* _imgA, const float* _imgB, int height, int width, std::string& imgType);
void PostCorrectorSamplerFunction(const GPUDevice &_dev, const int* _idxList, float* _out, int nBatch, int patchHeight, int patchWidth, int dimSize, std::string& imgType, int bufIdx);
void PostCorrectorColorBuffersAllocateFunction(const GPUDevice &_dev,const float* _rand, const float* _deno, const float* _ref, int height, int width);
void PostCorrectorColorBuffersDeallocateFunction();
void PostCorrectorGbuffersAllocateFunction(const GPUDevice &_dev, const float* _albedo, const float* _normal, int height, int width);
void PostCorrectorGbuffersDeallocateFunction();

class PostCorrectorConfigureOp : public OpKernel {
 public:
  explicit PostCorrectorConfigureOp(OpKernelConstruction* context) : OpKernel(context) {
	  context->GetAttr("img_height", &imgHeight);	
	  context->GetAttr("img_width", &imgWidth);	
    context->GetAttr("batch_size", &batchSize);		  
	  context->GetAttr("patch_height", &patchHeight);	
	  context->GetAttr("patch_width", &patchWidth);	
	  context->GetAttr("stride_height", &strideHeight);	
	  context->GetAttr("stride_width", &strideWidth);	
	  context->GetAttr("win_size", &winSize);	
  }
  
  void Compute(OpKernelContext* context) override {    	
    PostCorrectorConfigureFunction(
      imgHeight, imgWidth, patchHeight, patchWidth, strideHeight, strideWidth, batchSize, winSize
    );
  }
  
 private:
  int imgWidth, imgHeight, patchWidth, patchHeight, strideWidth, strideHeight, batchSize, winSize;
};

class PostCorrectorAssignBuffersOp: public OpKernel {
 public:
  explicit PostCorrectorAssignBuffersOp(OpKernelConstruction* context) : OpKernel(context) {
	context->GetAttr("img_type", &imgType);
  }
  
  void Compute(OpKernelContext* context) override {    
	const Tensor& imgA = context->input(0);
	const Tensor& imgB = context->input(1);

    const TensorShape& img_shape = imgA.shape();    
	
    PostCorrectorAssignBuffersFunction(
      context->eigen_device<GPUDevice>(),
	    imgA.flat<float>().data(), imgB.flat<float>().data(), 
	    img_shape.dim_size(1), img_shape.dim_size(2),
	    imgType);
  }
  
  private:  
	std::string imgType;  
};

class PostCorrectorSamplerOp : public OpKernel {
 public:
  explicit PostCorrectorSamplerOp(OpKernelConstruction* context) : OpKernel(context) {
	context->GetAttr("img_type", &imgType);
	context->GetAttr("buf_idx", &bufIdx);
	context->GetAttr("batch_size", &batchSize);
	context->GetAttr("patch_height", &patchHeight);
	context->GetAttr("patch_width", &patchWidth);
	context->GetAttr("dim_size", &dimSize);
  }

  void Compute(OpKernelContext* context) override {        
	const Tensor& idx_list = context->input(0);
           
    TensorShape output_shape;
    output_shape.AddDim(batchSize);
    output_shape.AddDim(patchHeight);
    output_shape.AddDim(patchWidth);
    output_shape.AddDim(dimSize);

    Tensor* out_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
    auto out_mat = out_tensor->tensor<float, 4>();
	
    PostCorrectorSamplerFunction(
      context->eigen_device<GPUDevice>(),
	  idx_list.flat<int>().data(), out_mat.data(),
      batchSize, patchHeight, patchWidth, dimSize, 
      imgType, bufIdx
      );
  }
  
  private:
    int batchSize, patchHeight, patchWidth, dimSize;
	std::string imgType;
	int bufIdx;
};

class PostCorrectorColorBuffersAllocateOp : public OpKernel {
 public:
  explicit PostCorrectorColorBuffersAllocateOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {    
	const Tensor& rand = context->input(0);
	const Tensor& deno = context->input(1);
  const Tensor& ref = context->input(2);

    const TensorShape& img_shape = rand.shape();    
	
    PostCorrectorColorBuffersAllocateFunction(
      context->eigen_device<GPUDevice>(),
	    rand.flat<float>().data(), deno.flat<float>().data(), ref.flat<float>().data(), 
	    img_shape.dim_size(1), img_shape.dim_size(2));
  }
};

class PostCorrectorColorBuffersDeallocateOp : public OpKernel {
 public:
  explicit PostCorrectorColorBuffersDeallocateOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {    	
    PostCorrectorColorBuffersDeallocateFunction();
  }
};

class PostCorrectorGbuffersAllocateOp : public OpKernel {
 public:
  explicit PostCorrectorGbuffersAllocateOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {    
	  const Tensor& albedo = context->input(0);
  	const Tensor& normal = context->input(1);

    const TensorShape& img_shape = albedo.shape();    
	
    PostCorrectorGbuffersAllocateFunction(
      context->eigen_device<GPUDevice>(),
	    albedo.flat<float>().data(), normal.flat<float>().data(), 
	    img_shape.dim_size(1), img_shape.dim_size(2));
  }
};

class PostCorrectorGbuffersDeallocateOp : public OpKernel {
 public:
  explicit PostCorrectorGbuffersDeallocateOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {    	
    PostCorrectorGbuffersDeallocateFunction();
  }
};


// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("PostCorrectorConfigure").Device(DEVICE_GPU), PostCorrectorConfigureOp);
REGISTER_KERNEL_BUILDER(Name("PostCorrectorAssignBuffers").Device(DEVICE_GPU), PostCorrectorAssignBuffersOp);
REGISTER_KERNEL_BUILDER(Name("PostCorrectorSampler").Device(DEVICE_GPU), PostCorrectorSamplerOp);
REGISTER_KERNEL_BUILDER(Name("PostCorrectorColorBuffersAllocate").Device(DEVICE_GPU), PostCorrectorColorBuffersAllocateOp);
REGISTER_KERNEL_BUILDER(Name("PostCorrectorColorBuffersDeallocate").Device(DEVICE_GPU), PostCorrectorColorBuffersDeallocateOp);
REGISTER_KERNEL_BUILDER(Name("PostCorrectorGbuffersAllocate").Device(DEVICE_GPU), PostCorrectorGbuffersAllocateOp);
REGISTER_KERNEL_BUILDER(Name("PostCorrectorGbuffersDeallocate").Device(DEVICE_GPU), PostCorrectorGbuffersDeallocateOp);
// ===========================================================


// ===========================================================
//    Post-correction function w/ cross-bilateral weighting
//        (Eq. 10 in paper)           (Eq. 11 in paper)
// ===========================================================
REGISTER_OP("PostCorrectorRevisedKernel")
  .Input("idx_list: int32")
  .Input("slope: float32")  
  .Input("wgt: float32")  
  .Input("alpha: float32")  
  .Output("out: float32")
  .Attr("buf_idx: int = 0")
  .Attr("training: bool")  
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle wgt_shape;	
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &wgt_shape));	
		
    std::vector<shape_inference::DimensionHandle> dims;
    dims.emplace_back(c->MakeDim(c->Dim(wgt_shape,0)));
    dims.emplace_back(c->MakeDim(c->Dim(wgt_shape,1)));
    dims.emplace_back(c->MakeDim(c->Dim(wgt_shape,2)));	
    dims.emplace_back(c->MakeDim(3));
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  });

REGISTER_OP("PostCorrectorRevisedKernelGrad")
  .Input("grad: float32")
  .Input("idx_list: int32")
  .Input("slope: float32")
  .Input("wgt: float32")
  .Input("alpha: float32")  
  .Output("grad_slope: float32")
  .Output("grad_wgt: float32")
  .Output("grad_alpha: float32")
  .Attr("buf_idx: int = 0")
  .Attr("training: bool")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(2));
	c->set_output(1, c->input(3));
  c->set_output(2, c->input(4));
    return Status::OK();
  });

void PostCorrectorRevisedKernelFunction(const GPUDevice &_dev, const int* _idxList, const float* _slope, const float* _wgt, const float* _alpha, float* _out, 
  int nBatch, int heightPatch, int widthPatch, int wgtDim, int bufIdx, bool training);
void PostCorrectorRevisedKernelGradFunction(const GPUDevice &_dev, const float* _inGrad, const int* _idxList, const float* _slope, const float* _wgt, const float* _alpha, 
  float* _outGradSlope, float* _outGradWgt, float* _outGradAlpha, int nBatch, int heightPatch, int widthPatch, int wgtDim, int bufIdx, bool training);

class PostCorrectorRevisedKernelOp : public OpKernel {
  public:
  explicit PostCorrectorRevisedKernelOp(OpKernelConstruction* context) : OpKernel(context) {	  
  	context->GetAttr("buf_idx", &bufIdx);
	  context->GetAttr("training", &training);	
  }

  void Compute(OpKernelContext* context) override {
  	const Tensor& idx_list = context->input(0);
  	const Tensor& slope = context->input(1);
    const Tensor& wgt = context->input(2);
    const Tensor& alpha = context->input(3);

    const TensorShape& wgt_shape = wgt.shape();    
    
    TensorShape output_shape;
    output_shape.AddDim(wgt_shape.dim_size(0));
    output_shape.AddDim(wgt_shape.dim_size(1));
    output_shape.AddDim(wgt_shape.dim_size(2));
    output_shape.AddDim(3);

    Tensor *out_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &out_tensor));
    auto out_mat = out_tensor->tensor<float, 4>();

    PostCorrectorRevisedKernelFunction(
      context->eigen_device<GPUDevice>(),
	    idx_list.flat<int>().data(), slope.flat<float>().data(), wgt.flat<float>().data(), alpha.flat<float>().data(), out_mat.data(), 
      output_shape.dim_size(0),
      output_shape.dim_size(1),
      output_shape.dim_size(2),
      wgt_shape.dim_size(3),
	    bufIdx, training);
  }

  private:
	  int bufIdx;
	  bool training;		
};

class PostCorrectorRevisedKernelGradOp : public OpKernel {
public:
  explicit PostCorrectorRevisedKernelGradOp(OpKernelConstruction* context) : OpKernel(context) {  
  	context->GetAttr("buf_idx", &bufIdx);
	  context->GetAttr("training", &training);
  }
  
  void Compute(OpKernelContext* context) override {
	  OP_REQUIRES(context, training == true, errors::InvalidArgument("PostCorrectorRevisedKernelGradOp is called in a test stage"));
    const Tensor& grad = context->input(0);
  	const Tensor& idx_list = context->input(1);
    const Tensor& slope = context->input(2);
  	const Tensor& wgt = context->input(3);
    const Tensor& alpha = context->input(4);
    
    const TensorShape& slope_shape = slope.shape();    
    const TensorShape& wgt_shape = wgt.shape();    
    const TensorShape& alpha_shape = alpha.shape();    
 
    Tensor* out_grad_slope_tensor = NULL;
    Tensor* out_grad_wgt_tensor = NULL;
    Tensor* out_grad_alpha_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, slope_shape, &out_grad_slope_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, wgt_shape, &out_grad_wgt_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, alpha_shape, &out_grad_alpha_tensor));
	
    auto out_grad_slope_mat = out_grad_slope_tensor->tensor<float, 4>();
	  auto out_grad_wgt_mat = out_grad_wgt_tensor->tensor<float, 4>();
    auto out_grad_alpha_mat = out_grad_alpha_tensor->tensor<float, 4>();

    PostCorrectorRevisedKernelGradFunction(
      context->eigen_device<GPUDevice>(), grad.flat<float>().data(), 
	    idx_list.flat<int>().data(), slope.flat<float>().data(), wgt.flat<float>().data(), alpha.flat<float>().data(), 
      out_grad_slope_mat.data(), out_grad_wgt_mat.data(), out_grad_alpha_mat.data(), 
      wgt_shape.dim_size(0),
	    wgt_shape.dim_size(1),
	    wgt_shape.dim_size(2),
      wgt_shape.dim_size(3),
	    bufIdx, training);
  }

  private:
  	int bufIdx;  
	  bool training;
};

REGISTER_KERNEL_BUILDER(Name("PostCorrectorRevisedKernel").Device(DEVICE_GPU), PostCorrectorRevisedKernelOp);
REGISTER_KERNEL_BUILDER(Name("PostCorrectorRevisedKernelGrad").Device(DEVICE_GPU), PostCorrectorRevisedKernelGradOp);
// ===========================================================
