#include <CUDA/_CUDA_NBody_Common.h>
#include <cstdlib>
#include <new>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <CUDA/helper_math.h>

#define __launch_bounds__(...) \
        __annotate__(launch_bounds(__VA_ARGS__))

#define dt 0.0005f

//__constant__ float dt;
//__constant__ float G;
//__constant__ unsigned int num;
__global__ __launch_bounds__(1024) void positionCalc(NBodyCUDAParticle* particles)
{
	unsigned int id = threadIdx.x + blockIdx.x * 1024;
	particles[id].position += particles[id].velocity * dt;
}
//__global__ void velocityCalc(NBodyCUDAParticle* particles)
//{
//	unsigned int id = threadIdx.x + blockIdx.x * 1024;
//	unsigned int c0 = 0;
//	float3 r = particles[id].position;
//	float3 dv = make_float3(0);
//	for (; c0 < 20 * 1024; ++c0)
//	{
//		float3 dr = particles[c0].position - r;
//		dv += (particles[c0].mass / (powf(dot(dr, dr), 1.5) + 0.00001)) * dr;
//	}
//	particles[id].velocity += dv * 0.001f * 0.005f;
//}
struct PosM
{
	float3 position;
	float mass;
};
__global__ void velocityCalc_Optimize1(NBodyCUDAParticle* particles)
{
	__shared__ PosM posM[1024];
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	float3 r = particles[id].position;
	float3 dv = make_float3(0);
	for (int c0(0); c0 < gridDim.x; ++c0)
	{
		posM[threadIdx.x] = *(PosM*)(particles + threadIdx.x + c0 * blockDim.x);
		__syncthreads();
		for (int c1(0); c1 < blockDim.x; ++c1)
		{
			float3 dr = posM[c1].position - r;
			float drr = rsqrtf(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + 0.00001f);
			// float drr = rsqrtf(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + 0.00000001f);
			drr = drr * drr * drr;
			dv += (posM[c1].mass * drr) * dr;
		}
		__syncthreads();
	}
	particles[id].velocity += dv * 0.001f * dt;
}
__global__ void forceCalc(NBodyCUDAParticle* particles, ExpData* expData)
{
	__shared__ PosM posM[1024];
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	float3 r = particles[id].position;
	float3 force = make_float3(0);
	for (int c0(0); c0 < gridDim.x; ++c0)
	{
		posM[threadIdx.x] = *(PosM*)(particles + threadIdx.x + c0 * blockDim.x);
		__syncthreads();
		for (int c1(0); c1 < blockDim.x; ++c1)
		{
			float3 dr = posM[c1].position - r;
			float drr = rsqrtf(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + 0.001f);
			drr = drr * drr * drr;
			force += (posM[c1].mass * drr) * dr;
		}
		__syncthreads();
	}
	float a(sqrtf(r.x * r.x + r.y * r.y + r.z * r.z));
	float fr = dot(force, r) / a;
	expData[id] = { a,fr };
}
#undef dt
NBodyCUDA_Glue::NBodyCUDA_Glue(unsigned int _blocks, float _dt, float _G)
	:
	blocks(_blocks),
	stream(nullptr)
{
	unsigned int _num(1024 * _blocks);
	cudaStreamCreate(&stream);
	//cudaMemcpyToSymbol(&dt, &_dt, sizeof(float));
	//cudaMemcpyToSymbol(&G, &_G, sizeof(float));
	//cudaMemcpyToSymbol(&num, &_num, sizeof(unsigned int));
}
void NBodyCUDA_Glue::run()
{
	positionCalc << < dim3(blocks, 1, 1), dim3(1024, 1, 1), 0, stream >> > (particles);
	velocityCalc_Optimize1 << < dim3(blocks, 1, 1), dim3(1024, 1, 1), 0, stream >> > (particles);
	cudaStreamSynchronize(stream);
}
void NBodyCUDA_Glue::experiment(ExpData* expData)
{
	forceCalc << < dim3(blocks, 1, 1), dim3(1024, 1, 1), 0, stream >> > (particles, expData);
	cudaStreamSynchronize(stream);
}