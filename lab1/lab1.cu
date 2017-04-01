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

#include "lab1.h"
static const unsigned W = 480;
static const unsigned H = 480;
static const unsigned NFRAME = 480;
static const unsigned VIDEO_BLOCK_MAPPING_SIZE = 4;
static const unsigned MAX_CELLS = W / VIDEO_BLOCK_MAPPING_SIZE;

__global__ void nextStatus(uint8_t* cells, uint8_t* rules, uint8_t rowIndex)
{
	uint8_t* currentRow = cells + MAX_CELLS*rowIndex;

	uint32_t colIndex = threadIdx.x;

	int32_t i1 = colIndex - 1;
	int32_t i2 = colIndex;
	int32_t i3 = colIndex + 1;

	if (i1 < 0)
	{
		i1 = MAX_CELLS - 1;
	}

	if (i3 >= MAX_CELLS)
	{
		i3 -= MAX_CELLS;
	}

	uint8_t b = 4 * currentRow[ i1] + 2 * currentRow[ i2] + currentRow[ i3];
	uint8_t* nextRow = currentRow + MAX_CELLS;

	nextRow[colIndex] = ((1 == rules[7 - b]) ? 1 : 0);
}

__global__ void nextFrame(uint8_t* frameBuffer,uint8_t* cells)
{
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	frameBuffer[y*gridDim.y+x] = cells[y / VIDEO_BLOCK_MAPPING_SIZE * MAX_CELLS  + x / VIDEO_BLOCK_MAPPING_SIZE] == 1 ? 255 : 0;
}

struct Lab1VideoGenerator::Impl
{
	int t = 0;
	uint8_t* cells;
	uint8_t* rules;
};

uint8_t* getRule(int x)
{
	uint8_t* bin = new uint8_t[8];
	uint8_t mask = 0b10000000;
	for (int i = 0; i < 8; ++i)
	{
		bin[i] = (x&mask) >> (7 - i);
		mask >>= 1;
	}
	return bin;
}

Lab1VideoGenerator::Lab1VideoGenerator() : impl(new Impl)
{
	cudaMalloc(&(this->impl->rules),8);
	cudaMemcpy(this->impl->rules, getRule(30), 8, cudaMemcpyHostToDevice);

	cudaMalloc(&(this->impl->cells), sizeof(uint8_t) * H / VIDEO_BLOCK_MAPPING_SIZE * W / VIDEO_BLOCK_MAPPING_SIZE);
	cudaMemset(this->impl->cells+MAX_CELLS/2,1,1);
}

Lab1VideoGenerator::~Lab1VideoGenerator()
{
}

void Lab1VideoGenerator::get_info(Lab1VideoInfo& info)
{
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 400;
	info.fps_d = 1;
}

void Lab1VideoGenerator::Generate(uint8_t* yuv)
{
	uint8_t rowIndex = this->impl->t / VIDEO_BLOCK_MAPPING_SIZE;
	nextStatus KERNEL_ARGS2(1, MAX_CELLS)(this->impl->cells, this->impl->rules, rowIndex);

	nextFrame KERNEL_ARGS2(dim3(W,H), dim3(1,1))(yuv, this->impl->cells);

	//cudaMemset(yuv, (impl->t) * 255 / NFRAME, W * H);
	cudaMemset(yuv + W * H, 128, W * H / 2);

	++(impl->t);
}