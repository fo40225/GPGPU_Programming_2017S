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

#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct transFunc : public thrust::unary_function<char, int>
{
	__device__ int operator()(char c) const
	{
		return c == '\n' ? 0 : 1;
	}
};

void CountPosition1(const char* text, int* pos, int text_size)
{
	auto ptrInput = thrust::device_pointer_cast(text);
	auto ptrOutput = thrust::device_pointer_cast(pos);

	//auto transFunc = []__device__(auto c) { return c == '\n' ? 0 : 1; };
	//auto transFunc = []__device__(auto c) { return (c ^ 0b1010) ? 1 : 0; };
	//auto transFunc = []__device__(auto c) { return (c ^ 0b1010) && 1; };
	//auto transFunc = []__device__(auto c) { return !!(c ^ 0b1010); };

	auto transBegin = thrust::make_transform_iterator(ptrInput, transFunc());

	//thrust::transform_iterator<decltype(transFunc), decltype(ptrInput), int> transBegin(ptrInput, transFunc);

	thrust::inclusive_scan_by_key(transBegin, transBegin + text_size, transBegin, ptrOutput);
}

__global__ void Count(const char* text, int* pos, int upperBound)
{
	auto offset = blockIdx.x * blockDim.x + threadIdx.x;

	if (upperBound < offset)
	{
		return;
	}

	if ('\n' == text[offset])
	{
		return;
	}

	if (0 == offset || '\n' == text[offset - 1])
	{
		auto loc = 1;

	loop:
		pos[offset] = loc++;

		if (++offset > upperBound || '\n' == text[offset])
		{
			return;
		}

		goto loop;
	}
}

void CountPosition2(const char* text, int* pos, int text_size)
{
	Count KERNEL_ARGS2(CeilDiv(text_size, 512), 512)(text, pos, text_size - 1);
}
