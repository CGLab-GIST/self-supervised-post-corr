
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


#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "cuda/include/cuda_runtime.h"
#include "cuda/include/vector_types.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

struct ConfPostCorrector {
	int imgWidth, imgHeight;
	int patchHeight, patchWidth;
	int strideHeight, strideWidth;
	int batchSize, winSize;	
};

#define SIZE_SUB_BUF 		2
#define SIZE_BUF 			(1 + SIZE_SUB_BUF)
#define NUM_EPSILON_DIV		1e-5f

// JH: it should be matched with NETOUT_SLOPE_NUM and NETOUT_WGT_DIM in main.py.
#define NETOUT_SLOPE_NUM		3
#define NETOUT_WGT_DIM			5

ConfPostCorrector gConf;
float4 *g_tmpCol = NULL;
cudaArray *g_src_rand[SIZE_BUF], *g_src_albedo[SIZE_BUF], *g_src_normal[SIZE_BUF];
cudaArray *g_src_deno[SIZE_BUF];
cudaArray *g_src_ref;
cudaTextureObject_t g_rand[SIZE_BUF], g_albedo[SIZE_BUF], g_normal[SIZE_BUF];
cudaTextureObject_t g_deno[SIZE_BUF];
cudaTextureObject_t g_ref;

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

__forceinline__ __host__ __device__ int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

__forceinline__ __host__ __device__ float4 operator+(float b, float4 a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__forceinline__ __host__ __device__ void operator-=(float4 &a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__forceinline__ __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__forceinline__ __device__ float norm2(const float4& a) {
	return (a.x * a.x + a.y * a.y + a.z * a.z);
}

__forceinline__ __device__ float norm2withLog(const float4& a) {
	return __logf(1.f + a.x * a.x + a.y * a.y + a.z * a.z);
}

__forceinline__ __device__ float2 getNormCoord(int x, int y, int width, int height) {
	return make_float2(((float)x + 0.5f) / (float)width, ((float)y + 0.5f) / (float)height);	
}

__forceinline__ __device__ float4 fetchTexVal(cudaTextureObject_t texImg, int x, int y, int width, int height) {
	float2 cCoord = getNormCoord(x, y, width, height);
	return tex2D<float4>(texImg, cCoord.x, cCoord.y);
}

__forceinline__ __device__ float Dot(const float4& a, const float4& b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

// Compute left-upper corner of the patches (reflection padding)
__forceinline__ __device__ int2 getCornerCoords(int patchIdx, int widthImg, int heightImg, int widthStride, int heightStride) {				
	int nPatchX = iDivUp(widthImg, widthStride);
	int nPatchY = iDivUp(heightImg, heightStride);
	int patchX = patchIdx % nPatchX;
	int patchY = patchIdx / nPatchX;
	int sx = (widthImg - nPatchX * widthStride) / 2;
	int sy = (heightImg - nPatchY * heightStride) / 2;		
	return make_int2(sx + patchX * widthStride, sy + patchY * heightStride);	
}

__global__ void LoadImages(float4* _outImg, const float* _img, int height, int width) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;				
	_outImg[cIdx] = make_float4(_img[cIdx * 3 + 0], _img[cIdx * 3 + 1], _img[cIdx * 3 + 2], 0.f);
}

__global__ void LoadImages4D(float4* _outImg, const float* _img, int height, int width) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;				
	_outImg[cIdx] = make_float4(_img[cIdx * 4 + 0], _img[cIdx * 4 + 1], _img[cIdx * 4 + 2], _img[cIdx * 4 + 3]);
}

__global__ void ExtractPatches(float* _out, const int* _idxArr, int iBatch, 
							   int patchHeight, int patchWidth, int strideHeight, int strideWidth, int dimSize, 
							   cudaTextureObject_t texImg, int height, int width) {	
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= patchWidth || cy >= patchHeight)
		return;	

	int2 cornerCoords = getCornerCoords(_idxArr[iBatch], width, height, strideWidth, strideHeight);		
	const float4& inCol = fetchTexVal(texImg, cornerCoords.x + cx, cornerCoords.y + cy, width, height);		
	int sidx = iBatch * patchWidth * patchHeight * dimSize + (cy * patchWidth + cx) * dimSize;			
	_out[sidx + 0] = inCol.x;
	_out[sidx + 1] = inCol.y;
	_out[sidx + 2] = inCol.z;		
}

__global__ void ExtractPatches4D(float* _out, const int* _idxArr, int iBatch, 
							   int patchHeight, int patchWidth, int strideHeight, int strideWidth, int dimSize, 
							   cudaTextureObject_t texImg, int height, int width) {	
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= patchWidth || cy >= patchHeight)
		return;	

	int2 cornerCoords = getCornerCoords(_idxArr[iBatch], width, height, strideWidth, strideHeight);		
	const float4& inCol = fetchTexVal(texImg, cornerCoords.x + cx, cornerCoords.y + cy, width, height);		
	int sidx = iBatch * patchWidth * patchHeight * dimSize + (cy * patchWidth + cx) * dimSize;			
	_out[sidx + 0] = inCol.x;
	_out[sidx + 1] = inCol.y;
	_out[sidx + 2] = inCol.z;
	_out[sidx + 3] = inCol.w;
}

void PostCorrectorConfigureFunction(int imgHeight, int imgWidth, int patchHeight, int patchWidth, int strideHeight, int strideWidth, int batchSize, int winSize) {
	gConf.imgWidth = imgWidth;
	gConf.imgHeight = imgHeight;
	gConf.patchHeight = patchHeight;
	gConf.patchWidth = patchWidth;
	gConf.strideHeight = strideHeight;
	gConf.strideWidth = strideWidth;
	gConf.batchSize = batchSize;
	gConf.winSize = winSize;
}

// Currently support for only two buffers
void PostCorrectorAssignBuffersFunction(const GPUDevice &_dev, const float* _imgA, const float* _imgB, int height, int width, std::string& imgType) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));
	
	if (imgType == "rand") {
		LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgA, height, width);	
		cudaMemcpy2DToArray(g_src_rand[1], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);			
		LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgB, height, width);	
		cudaMemcpy2DToArray(g_src_rand[2], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);			
	}
	// albedo w/ visibility
	else if (imgType == "albedo") {
		LoadImages4D <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgA, height, width);	
		cudaMemcpy2DToArray(g_src_albedo[1], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);			
		LoadImages4D <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgB, height, width);	
		cudaMemcpy2DToArray(g_src_albedo[2], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);			
	}
	else if (imgType == "normal") {
		LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgA, height, width);	
		cudaMemcpy2DToArray(g_src_normal[1], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);			
		LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgB, height, width);	
		cudaMemcpy2DToArray(g_src_normal[2], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);		
	}
	else if (imgType == "deno") {
		LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgA, height, width);	
		cudaMemcpy2DToArray(g_src_deno[1], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);			
		LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _imgB, height, width);	
		cudaMemcpy2DToArray(g_src_deno[2], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);			
	}
}

void PostCorrectorColorBuffersAllocateFunction(const GPUDevice &_dev,const float* _rand, const float* _deno, const float* _ref, int height, int width) { 
	cudaMalloc((void **)&g_tmpCol, width * height * sizeof(float4)); 

	cudaChannelFormatDesc channelDescCol = cudaCreateChannelDesc<float4>();
	for (int buf = 0; buf < SIZE_BUF; ++buf) {
		cudaMallocArray(&g_src_rand[buf], &channelDescCol, width, height);	
		cudaMallocArray(&g_src_deno[buf], &channelDescCol, width, height);
	}	
	cudaMallocArray(&g_src_ref, &channelDescCol, width, height);
  
	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeMirror;
	texDescr.addressMode[1] = cudaAddressModeMirror;
	texDescr.readMode = cudaReadModeElementType;

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;		
	for (int buf = 0; buf < SIZE_BUF; ++buf) {	
		texRes.res.array.array = g_src_rand[buf];
		cudaCreateTextureObject(&g_rand[buf], &texRes, &texDescr, NULL);			
		texRes.res.array.array = g_src_deno[buf];
		cudaCreateTextureObject(&g_deno[buf], &texRes, &texDescr, NULL);	
	}	

	texRes.res.array.array = g_src_ref;
	cudaCreateTextureObject(&g_ref, &texRes, &texDescr, NULL);	
	
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));
	
	// Full-buffers
	// noisy image
    LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _rand, height, width);	
	cudaMemcpy2DToArray(g_src_rand[0], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);	
	// denoised image
    LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _deno, height, width);	
	cudaMemcpy2DToArray(g_src_deno[0], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);
	// reference for debugging purpose
    LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _ref, height, width);	
	cudaMemcpy2DToArray(g_src_ref, 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);
}

void PostCorrectorGbuffersAllocateFunction(const GPUDevice &_dev, const float* _albedo, const float* _normal, int height, int width) {
	cudaChannelFormatDesc channelDescCol = cudaCreateChannelDesc<float4>();
	for (int buf = 0; buf < SIZE_BUF; ++buf) {
		cudaMallocArray(&g_src_albedo[buf], &channelDescCol, width, height);	
		cudaMallocArray(&g_src_normal[buf], &channelDescCol, width, height);
	}	
  
	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeMirror;
	texDescr.addressMode[1] = cudaAddressModeMirror;
	texDescr.readMode = cudaReadModeElementType;

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;		
	for (int buf = 0; buf < SIZE_BUF; ++buf) {	
		texRes.res.array.array = g_src_albedo[buf];
		cudaCreateTextureObject(&g_albedo[buf], &texRes, &texDescr, NULL);	
		texRes.res.array.array = g_src_normal[buf];
		cudaCreateTextureObject(&g_normal[buf], &texRes, &texDescr, NULL);	
	}	
	
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));
	
	// Full-buffers
	// albedo (w/ visibility)
    LoadImages4D <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _albedo, height, width);	
	cudaMemcpy2DToArray(g_src_albedo[0], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);
	// normal
    LoadImages <<< grid, threads, 0, _dev.stream() >>> (g_tmpCol, _normal, height, width);	
	cudaMemcpy2DToArray(g_src_normal[0], 0, 0, g_tmpCol, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);
}

void PostCorrectorGbuffersDeallocateFunction() {
	for (int buf = 0; buf < SIZE_BUF; ++buf) {
		cudaDestroyTextureObject(g_albedo[buf]);
		cudaDestroyTextureObject(g_normal[buf]);
	}	
	
	for (int buf = 0; buf < SIZE_BUF; ++buf) {
		cudaFreeArray(g_src_albedo[buf]);
		cudaFreeArray(g_src_normal[buf]);
	}		
}

void PostCorrectorColorBuffersDeallocateFunction() {
	for (int buf = 0; buf < SIZE_BUF; ++buf) {
		cudaDestroyTextureObject(g_rand[buf]);	
		cudaDestroyTextureObject(g_deno[buf]);
	}	
	cudaDestroyTextureObject(g_ref);	

	for (int buf = 0; buf < SIZE_BUF; ++buf) {
		cudaFreeArray(g_src_rand[buf]);
		cudaFreeArray(g_src_deno[buf]);
	}	

	cudaFreeArray(g_src_ref);	
	cudaFree(g_tmpCol);
}

void PostCorrectorSamplerFunction(const GPUDevice &_dev, const int* _idxList, float* _out, int nBatch, int patchHeight, int patchWidth, int dimSize, std::string& imgType, int bufIdx) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(patchWidth, blockDim), iDivUp(patchHeight, blockDim));

	cudaTextureObject_t texImg;
	
	if (imgType == "rand")
		texImg = g_rand[bufIdx];
	else if (imgType == "albedo")
		texImg = g_albedo[bufIdx];	
	else if (imgType == "normal")
		texImg = g_normal[bufIdx];
	else if (imgType == "ref")
		texImg = g_ref;
	else if (imgType == "deno")
		texImg = g_deno[bufIdx];
	for (int iBatch = 0; iBatch < nBatch; ++iBatch) {
		if (dimSize == 4) {
			ExtractPatches4D<<< grid, threads, 0, _dev.stream() >>>  (_out, _idxList, iBatch, 
																	  patchHeight, patchWidth, gConf.strideHeight, gConf.strideWidth, dimSize,
																	  texImg, gConf.imgHeight, gConf.imgWidth);	
		}
		if (dimSize == 3) {
			ExtractPatches<<< grid, threads, 0, _dev.stream() >>>  (_out, _idxList, iBatch, 
																	patchHeight, patchWidth, gConf.strideHeight, gConf.strideWidth, dimSize,
																	texImg, gConf.imgHeight, gConf.imgWidth);	
		}
	}
}

// ===========================================================
//    Post-correction function w/ cross-bilateral weighting
//        (Eq. 10 in paper)           (Eq. 11 in paper)
// ===========================================================

__forceinline__ __device__ float KernelCombRevised(const float* stdDev, 
	const float4& dAlbedo, const float4& dNormal, const float4& dRand, const float4& dDeno) {
	float dist2_albedo = norm2(dAlbedo) * (stdDev[0]);
	float dist2_normal = norm2(dNormal) * (stdDev[1]);
	float dist2_vis = (dAlbedo.w * dAlbedo.w) * (stdDev[2]);
	float dist2_rand = norm2withLog(dRand) * (stdDev[3]);
	float dist2_deno = norm2withLog(dDeno) * (stdDev[4]);
	float w = __expf(-dist2_albedo-dist2_normal-dist2_vis-dist2_rand-dist2_deno);
	return w;
}

__forceinline__ __device__ void GradKernelCombRevised(float *outGrad, const float* stdDev, 
	const float4& dAlbedo, const float4& dNormal, const float4& dRand, const float4& dDeno) {
	float dist2_albedo = norm2(dAlbedo) * (stdDev[0]);
	float dist2_normal = norm2(dNormal) * (stdDev[1]);
	float dist2_vis = (dAlbedo.w * dAlbedo.w) * (stdDev[2]);
	float dist2_rand = norm2withLog(dRand) * (stdDev[3]);
	float dist2_deno = norm2withLog(dDeno) * (stdDev[4]);
	float w = __expf(-dist2_albedo-dist2_normal-dist2_vis-dist2_rand-dist2_deno);
	
	outGrad[0] = -w * norm2(dAlbedo);
	outGrad[1] = -w * norm2(dNormal);
	outGrad[2] = -w * (dAlbedo.w * dAlbedo.w);
	outGrad[3] = -w * norm2withLog(dRand);
	outGrad[4] = -w * norm2withLog(dDeno);
}

__global__ void PostCorrectorRevisedKernelCUDA(const int* _idxArr, const float* _slope, const float* _wgt, const float* _alpha, float* _out, int iBatch, 
							   				  int heightPatch, int widthPatch, int heightImg, int widthImg, int heightStride, int widthStride, int winSize, int wgtDim, 
                               				  cudaTextureObject_t texRand, cudaTextureObject_t texDeno, cudaTextureObject_t texAlbedo, cudaTextureObject_t texNormal, bool training) {

	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);	
	if (cx >= widthPatch || cy >= heightPatch)
		return;
	
	int2 cornerCoords = make_int2(0, 0);
	if (training)
		cornerCoords = getCornerCoords(_idxArr[iBatch], widthImg, heightImg, widthStride, heightStride);

	const int cIdx = cy * widthPatch + cx;	  
	const int nPix = widthPatch * heightPatch;       
	const int halfWinSize = winSize / 2; 
	const int colorDim = 3;
	
	float4 cRand = fetchTexVal(texRand, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	float4 cDeno = fetchTexVal(texDeno, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	const float4& cAlbedo = fetchTexVal(texAlbedo, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	const float4& cNormal = fetchTexVal(texNormal, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	const float& cAlpha = _alpha[iBatch * nPix + cIdx];

	float4 cSlope[NETOUT_SLOPE_NUM];
	for (int i = 0; i < NETOUT_SLOPE_NUM; ++i)
		cSlope[i] = make_float4(_slope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 0], 
								_slope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 1], 
								_slope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 2], 0.f);	

	float cStdDev[NETOUT_WGT_DIM] = {0.f,};
	for (int wgtIdx = 0; wgtIdx < wgtDim; wgtIdx++) {
		cStdDev[wgtIdx] = _wgt[(iBatch * nPix + cIdx) * wgtDim + wgtIdx];
	}


	float sumW = 0.f;
	float4 cIntercept = make_float4(0.f, 0.f, 0.f, 0.f);		
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {	
			float4 iRand = fetchTexVal(texRand, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg);
			float4 dDeno = fetchTexVal(texDeno, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg) - cDeno;
			float4 dAlbedo = fetchTexVal(texAlbedo, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg) - cAlbedo;
			float4 dNormal = fetchTexVal(texNormal, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg) - cNormal;

			float weight = cAlpha;
			if (!(iy == cy && ix == cx)) {
				weight = KernelCombRevised(cStdDev, dAlbedo, dNormal, iRand-cRand, dDeno);
			}
			cIntercept += weight * (iRand - cSlope[0] * dDeno - cSlope[1] * dAlbedo - cSlope[2] * dNormal);
			sumW += weight;
		}
	}

	float invSumW = 1.f / fmaxf(sumW, NUM_EPSILON_DIV);
	cIntercept *= invSumW;
	
	_out[(iBatch * nPix + cIdx) * colorDim + 0] = cIntercept.x;
	_out[(iBatch * nPix + cIdx) * colorDim + 1] = cIntercept.y;
	_out[(iBatch * nPix + cIdx) * colorDim + 2] = cIntercept.z;
}

__global__ void PostCorrectorRevisedKernelGradCUDA(const float* _inGrad, const int* _idxArr, const float* _slope, const float* _wgt, const float* _alpha, float* _outGradSlope, float* _outGradWgt, float* _outGradAlpha, 
								   				  int iBatch, int heightPatch, int widthPatch, int heightImg, int widthImg, int heightStride, int widthStride, int winSize, int wgtDim, 
								   				  cudaTextureObject_t texRand, cudaTextureObject_t texDeno, cudaTextureObject_t texAlbedo, cudaTextureObject_t texNormal, bool training) { 
												
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= widthPatch || cy >= heightPatch)
		return;

	int2 cornerCoords = make_int2(0, 0);
	if (training)
		cornerCoords = getCornerCoords(_idxArr[iBatch], widthImg, heightImg, widthStride, heightStride);
	
	const int cIdx = cy * widthPatch + cx;		
	const int nPix = widthPatch * heightPatch;       
	const int halfWinSize = winSize / 2;
	const int colorDim = 3;
	
	const float4& cRand = fetchTexVal(texRand, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	const float4& cDeno = fetchTexVal(texDeno, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	const float4& cAlbedo = fetchTexVal(texAlbedo, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	const float4& cNormal = fetchTexVal(texNormal, cornerCoords.x + cx, cornerCoords.y + cy, widthImg, heightImg);
	const float4& cInGradCol = make_float4(_inGrad[(iBatch * nPix + cIdx) * colorDim + 0], 
										   _inGrad[(iBatch * nPix + cIdx) * colorDim + 1], 
										   _inGrad[(iBatch * nPix + cIdx) * colorDim + 2], 0.f);
	const float& cAlpha = _alpha[iBatch * nPix + cIdx];
	
	float4 cSlope[NETOUT_SLOPE_NUM];
	for (int i = 0; i < NETOUT_SLOPE_NUM; ++i)
		cSlope[i] = make_float4(_slope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 0], 
								_slope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 1], 
								_slope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 2], 0.f);	

	float cStdDev[NETOUT_WGT_DIM] = {0.f,};
	for (int wgtIdx = 0; wgtIdx < wgtDim; wgtIdx++) {
		cStdDev[wgtIdx] = _wgt[(iBatch * nPix + cIdx) * wgtDim + wgtIdx];
	}

	float ciGradBand[NETOUT_WGT_DIM] = {0.f,};
	float4 tmpOutGradWgtA[NETOUT_WGT_DIM];
	float tmpOutGradWgtB[NETOUT_WGT_DIM] = {0.f,};
	for (int wgtIdx = 0; wgtIdx < wgtDim; wgtIdx++) {
		tmpOutGradWgtA[wgtIdx] = make_float4(0.f, 0.f, 0.f, 0.f);
	}
	float4 tmpOutGradSlope[NETOUT_SLOPE_NUM];
	for (int i = 0; i < NETOUT_SLOPE_NUM; ++i)
		tmpOutGradSlope[i] = make_float4(0.f, 0.f, 0.f, 0.f);
	
	float sumW = 0.f;
	float4 cIntercept = make_float4(0.f, 0.f, 0.f, 0.f);		
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++) {
			const float4& iRand = fetchTexVal(texRand, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg);
			float4 dDeno = fetchTexVal(texDeno, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg) - cDeno;
			float4 dAlbedo = fetchTexVal(texAlbedo, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg) - cAlbedo;
			float4 dNormal = fetchTexVal(texNormal, cornerCoords.x + ix, cornerCoords.y + iy, widthImg, heightImg) - cNormal;

			// accumulating each values for gradients
			float weight = cAlpha;
			if (!(iy == cy && ix == cx)) {
				weight = KernelCombRevised(cStdDev, dAlbedo, dNormal, iRand-cRand, dDeno);
				GradKernelCombRevised(ciGradBand, cStdDev, dAlbedo, dNormal, iRand-cRand, dDeno);
				for (int i = 0; i < wgtDim; ++i) {
					tmpOutGradWgtA[i] += ciGradBand[i] * (iRand - cSlope[0] * dDeno - cSlope[1] * dAlbedo - cSlope[2] * dNormal);
					tmpOutGradWgtB[i] += ciGradBand[i];
				}
			}

			tmpOutGradSlope[0] += -weight * dDeno;
			tmpOutGradSlope[1] += -weight * dAlbedo;
			tmpOutGradSlope[2] += -weight * dNormal;
			
			cIntercept += weight * (iRand - cSlope[0] * dDeno - cSlope[1] * dAlbedo - cSlope[2] * dNormal);
			
			sumW += weight;
		}
	}

	float invSumW = 1.f / fmaxf(sumW, NUM_EPSILON_DIV);
	cIntercept *= invSumW;

	// saving gradient w.r.t. sigma and alpha
	for (int i = 0; i < wgtDim; ++i) {
		_outGradWgt[(iBatch * nPix + cIdx) * wgtDim + i] = invSumW * Dot(cInGradCol, tmpOutGradWgtA[i] - tmpOutGradWgtB[i] * cIntercept);
	}
	_outGradAlpha[iBatch * nPix + cIdx] = invSumW * Dot(cInGradCol, cRand - cIntercept);

	// saving gradient w.r.t. slope
	for (int i = 0; i < NETOUT_SLOPE_NUM; ++i) {
		_outGradSlope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 0] = invSumW * cInGradCol.x * tmpOutGradSlope[i].x;
		_outGradSlope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 1] = invSumW * cInGradCol.y * tmpOutGradSlope[i].y;
		_outGradSlope[NETOUT_SLOPE_NUM * colorDim * (iBatch * nPix + cIdx) + i * colorDim + 2] = invSumW * cInGradCol.z * tmpOutGradSlope[i].z;
	}
}

void PostCorrectorRevisedKernelFunction(const GPUDevice &_dev, const int* _idxList, const float* _slope, const float* _wgt, const float* _alpha, float* _out, 
  	int nBatch, int heightPatch, int widthPatch, int wgtDim, int bufIdx, bool training) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(widthPatch, blockDim), iDivUp(heightPatch, blockDim));

	for (int iBatch = 0; iBatch < nBatch; ++iBatch) 
		PostCorrectorRevisedKernelCUDA <<< grid, threads, 0, _dev.stream() >>> (_idxList, _slope, _wgt, _alpha, _out, iBatch, 
																  		 	    heightPatch, widthPatch, gConf.imgHeight, gConf.imgWidth, gConf.strideHeight, gConf.strideWidth, 
																		 	    gConf.winSize, wgtDim, g_rand[bufIdx], g_deno[bufIdx], g_albedo[bufIdx], g_normal[bufIdx], training);
}

void PostCorrectorRevisedKernelGradFunction(const GPUDevice &_dev, const float* _inGrad, const int* _idxList, const float* _slope, const float* _wgt, const float* _alpha, 
  	float* _outGradSlope, float* _outGradWgt, float* _outGradAlpha, int nBatch, int heightPatch, int widthPatch, int wgtDim, int bufIdx, bool training) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(widthPatch, blockDim), iDivUp(heightPatch, blockDim));

	for (int iBatch = 0; iBatch < nBatch; ++iBatch) 
		PostCorrectorRevisedKernelGradCUDA <<< grid, threads, 0, _dev.stream() >>> (_inGrad, _idxList, _slope, _wgt, _alpha, _outGradSlope, _outGradWgt, _outGradAlpha, iBatch, 
																	 			    heightPatch, widthPatch, gConf.imgHeight, gConf.imgWidth, gConf.strideHeight, gConf.strideWidth, 
																				    gConf.winSize, wgtDim, g_rand[bufIdx], g_deno[bufIdx], g_albedo[bufIdx], g_normal[bufIdx], training);
}
// ===========================================================

#endif  // GOOGLE_CUDA
