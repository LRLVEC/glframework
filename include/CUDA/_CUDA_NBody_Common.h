#pragma once
#include <cuda_runtime.h>
struct NBodyCUDAParticle
{
	float3 position;
	float mass;
	float3 velocity;
	float v;
};
struct ExpData
{
	float r;
	float force;
};
struct NBodyCUDA_Device;
struct NBodyCUDA_Glue
{
	NBodyCUDAParticle* particles;
	unsigned int blocks;
	NBodyCUDA_Glue(unsigned int _blocks, float _dt, float _G);
	void run();
	void experiment(ExpData* expData);
};