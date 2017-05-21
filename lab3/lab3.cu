#if defined  __INTELLISENSE__ || defined  __RESHARPER__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float* background,
	const float* target,
	const float* mask,
	float* output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f)
	{
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb * yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb)
		{
			output[curb * 3 + 0] = target[curt * 3 + 0];
			output[curb * 3 + 1] = target[curt * 3 + 1];
			output[curb * 3 + 2] = target[curt * 3 + 2];
		}
	}
}

__global__ void PoissonImageCloningIteration(
	const float* fixed,
	const float* mask,
	const float* buf1,
	float* buf2,
	const int wt, const int ht
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;

	const int curt = wt * yt + xt;

	if (yt < ht && xt < wt && mask[curt] > 127.0f)
	{
		auto idxNb = wt * (yt - 1) + xt;
		auto idxWb = wt * yt + (xt - 1);
		auto idxSb = wt * (yt + 1) + xt;
		auto idxEb = wt * yt + (xt + 1);

		auto rValue = fixed[curt * 3 + 0];
		auto gValue = fixed[curt * 3 + 1];
		auto bValue = fixed[curt * 3 + 2];

		if (0 != yt && mask[idxNb] > 127.0f)
		{
			rValue += buf1[idxNb * 3 + 0];
			gValue += buf1[idxNb * 3 + 1];
			bValue += buf1[idxNb * 3 + 2];
		}

		if (0 != xt && mask[idxWb] > 127.0f)
		{
			rValue += buf1[idxWb * 3 + 0];
			gValue += buf1[idxWb * 3 + 1];
			bValue += buf1[idxWb * 3 + 2];
		}

		if (ht != yt + 1 && mask[idxSb] > 127.0f)
		{
			rValue += buf1[idxSb * 3 + 0];
			gValue += buf1[idxSb * 3 + 1];
			bValue += buf1[idxSb * 3 + 2];
		}

		if (wt != xt + 1 && mask[idxEb] > 127.0f)
		{
			rValue += buf1[idxEb * 3 + 0];
			gValue += buf1[idxEb * 3 + 1];
			bValue += buf1[idxEb * 3 + 2];
		}

		buf2[curt * 3 + 0] = 1.0f / 4.0f * rValue;
		buf2[curt * 3 + 1] = 1.0f / 4.0f * gValue;
		buf2[curt * 3 + 2] = 1.0f / 4.0f * bValue;
	}
}

__global__ void CalculateFixed(
	const float* background,
	const float* target,
	const float* mask,
	float* fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;

	const int curt = wt * yt + xt;

	if (yt < ht && xt < wt && mask[curt] > 127.0f)
	{
		const int yb = oy + yt;
		const int xb = ox + xt;

		const int curb = wb * yb + xb;

		auto idxCt = curt;
		auto idxNt = 0 == yt ? curt : wt * (yt - 1) + xt;
		auto idxWt = 0 == xt ? curt : wt * yt + (xt - 1);
		auto idxSt = ht == yt + 1 ? curt : wt * (yt + 1) + xt;
		auto idxEt = wt == xt + 1 ? curt : wt * yt + (xt + 1);

		fixed[curt * 3 + 0] = 4.0f * target[idxCt * 3 + 0] - (target[idxNt * 3 + 0] + target[idxWt * 3 + 0] + target[idxSt * 3 + 0] + target[idxEt * 3 + 0]);
		fixed[curt * 3 + 1] = 4.0f * target[idxCt * 3 + 1] - (target[idxNt * 3 + 1] + target[idxWt * 3 + 1] + target[idxSt * 3 + 1] + target[idxEt * 3 + 1]);
		fixed[curt * 3 + 2] = 4.0f * target[idxCt * 3 + 2] - (target[idxNt * 3 + 2] + target[idxWt * 3 + 2] + target[idxSt * 3 + 2] + target[idxEt * 3 + 2]);

		auto idxNb = yb < 0 ? wb * 0 + xb : wb * (yb - 1) + xb;
		auto idxWb = xb < 0 ? wb * yb + 0 : wb * yb + (xb - 1);
		auto idxSb = hb < yb ? wb * yb + xb : wb * (yb + 1) + xb;
		auto idxEb = wb < xb ? wb * yb + xb : wb * yb + (xb + 1);

		if (0 == yt || mask[idxNt] <= 127.0f)
		{
			fixed[curt * 3 + 0] += background[idxNb * 3 + 0];
			fixed[curt * 3 + 1] += background[idxNb * 3 + 1];
			fixed[curt * 3 + 2] += background[idxNb * 3 + 2];
		}

		if (0 == xt || mask[idxWt] <= 127.0f)
		{
			fixed[curt * 3 + 0] += background[idxWb * 3 + 0];
			fixed[curt * 3 + 1] += background[idxWb * 3 + 1];
			fixed[curt * 3 + 2] += background[idxWb * 3 + 2];
		}

		if (ht == yt + 1 || mask[idxSt] <= 127.0f)
		{
			fixed[curt * 3 + 0] += background[idxSb * 3 + 0];
			fixed[curt * 3 + 1] += background[idxSb * 3 + 1];
			fixed[curt * 3 + 2] += background[idxSb * 3 + 2];
		}

		if (wt == xt + 1 || mask[idxEt] <= 127.0f)
		{
			fixed[curt * 3 + 0] += background[idxEb * 3 + 0];
			fixed[curt * 3 + 1] += background[idxEb * 3 + 1];
			fixed[curt * 3 + 2] += background[idxEb * 3 + 2];
		}
	}
}

void PoissonImageCloning(
	const float* background,
	const float* target,
	const float* mask,
	float* output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3 * wt * ht * sizeof(float));
	cudaMalloc(&buf1, 3 * wt * ht * sizeof(float));
	cudaMalloc(&buf2, 3 * wt * ht * sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	CalculateFixed KERNEL_ARGS2(gdim, bdim)(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
		);

	cudaMemcpy(buf1, target, sizeof(float) * 3 * wt * ht, cudaMemcpyDeviceToDevice);

	// iterate
	for (int i = 0; i < 10000; ++i)
	{
		PoissonImageCloningIteration KERNEL_ARGS2(gdim, bdim)(
			fixed, mask, buf1, buf2, wt, ht
			);

		PoissonImageCloningIteration KERNEL_ARGS2(gdim, bdim)(
			fixed, mask, buf2, buf1, wt, ht
			);
	}

	// copy the image back
	cudaMemcpy(output, background, wb * hb * sizeof(float) * 3, cudaMemcpyDeviceToDevice);
	SimpleClone KERNEL_ARGS2(gdim, bdim)(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
		);

	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}