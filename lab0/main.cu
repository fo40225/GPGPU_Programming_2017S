#if defined  __INTELLISENSE__ || defined  __RESHARPER__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

const int W = 40;
const int H = 12;

__global__ void Draw(char* frame)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H && x < W)
	{
		char c;
		if (x == W - 1)
		{
			c = y == H - 1 ? '\0' : '\n';
		}
		else if (y == 0 || y == H - 1 || x == 0 || x == W - 2)
		{
			c = ':';
		}
		else
		{
			c = ' ';
			if (y > (H / 3) && (x > ((W / 3) - 4) + ((H - y - 3) * (W / 20)) && x < (((W / 3) * 2) - 4)))
			{
				c = '#';
			}

			if (x == (W - (W / 10) - 1 - 2))
			{
				if (y == (H - 1) - 1)
				{
					c = '#';
				}
				else
				{
					if (y > (H / 3))
					{
						c = '|';
					}
				}
			}

			if (y == (H / 3) + 1 && x == (W - (W / 10) - 1 - 2 - 1))
			{
				c = '<';
			}
		}

		frame[y * W + x] = c;
	}
}

int main(int argc, char** argv)
{
	MemoryBuffer<char> frame(W * H);
	auto frame_smem = frame.CreateSync(W * H);
	CHECK;

	Draw<<<dim3((W - 1) / 16 + 1, (H - 1) / 12 + 1), dim3(16, 12)>>>(frame_smem.get_gpu_wo());
	CHECK;

	puts(frame_smem.get_cpu_ro());
	CHECK;
	return 0;
}