#pragma once
#include <cuda_runtime.h>

#include <iostream>

#ifndef CUDA_CHECK_THROW
#define NotDefinedCUDA_CHECK_THROW
#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)\
do {\
	cudaError_t result = x;\
	if (result != cudaSuccess)\
		throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result));\
} while(0)
#define CUDA_CHECK_PRINT(x)\
do {\
	cudaError_t result = x;\
	if (result != cudaSuccess)\
		std::cout << FILE_LINE " " #x " failed with error " << cudaGetErrorString(result) << std::endl;\
} while(0)
#endif

namespace CUDA
{
	// pure cuda texture created from cuda array
	template<unsigned int dim>struct Texture
	{
		static_assert(dim&& dim < 4, "Dim must be one of 1, 2, 3!");
	};

	template<>struct Texture<1>
	{
		cudaArray* array;
		cudaTextureObject_t texture;

		Texture(size_t width,
			cudaChannelFormatDesc const& channelDesc,
			void const* src = nullptr,
			cudaTextureAddressMode addressMode = cudaAddressModeClamp,
			cudaTextureFilterMode filterMode = cudaFilterModeLinear,
			cudaTextureReadMode readMode = cudaReadModeNormalizedFloat,
			bool normalizedCoords = true)
			:
			array(nullptr),
			texture(0)
		{
			if (width)
			{
				CUDA_CHECK_THROW(cudaMallocArray(&array, &channelDesc, width));
				if (src)
				{
					size_t element_size = (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w) / 8;
					CUDA_CHECK_THROW(cudaMemcpy2DToArray(array, 0, 0, src, element_size * width, element_size * width, 1, cudaMemcpyHostToDevice));
				}
				cudaResourceDesc resDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = array;
				cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = addressMode;
				texDesc.filterMode = filterMode;
				texDesc.readMode = readMode;
				texDesc.normalizedCoords = normalizedCoords;
				CUDA_CHECK_PRINT(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
			}
		}

		~Texture()
		{
			if (texture)
			{
				CUDA_CHECK_PRINT(cudaDestroyTextureObject(texture));
				texture = 0;
			}
			if (array)
			{
				CUDA_CHECK_PRINT(cudaFreeArray(array));
				array = nullptr;
			}
		}
		operator cudaTextureObject_t()const
		{
			return texture;
		}
	};

	template<>struct Texture<2>
	{
		cudaArray* array;
		cudaTextureObject_t texture;

		Texture() :array(nullptr), texture(0) {}

		Texture(size_t width, size_t height,
			cudaChannelFormatDesc const& channelDesc,
			void const* src = nullptr,
			cudaTextureAddressMode addressMode = cudaAddressModeClamp,
			cudaTextureFilterMode filterMode = cudaFilterModeLinear,
			cudaTextureReadMode readMode = cudaReadModeNormalizedFloat,
			bool normalizedCoords = true)
			: Texture()
		{
			create(width, height, channelDesc, src, addressMode, filterMode, readMode, normalizedCoords);
		}

		~Texture()
		{
			if (texture)
			{
				CUDA_CHECK_PRINT(cudaDestroyTextureObject(texture));
				texture = 0;
			}
			if (array)
			{
				CUDA_CHECK_PRINT(cudaFreeArray(array));
				array = nullptr;
			}
		}

		void create(size_t width, size_t height,
			cudaChannelFormatDesc const& channelDesc,
			void const* src = nullptr,
			cudaTextureAddressMode addressMode = cudaAddressModeClamp,
			cudaTextureFilterMode filterMode = cudaFilterModeLinear,
			cudaTextureReadMode readMode = cudaReadModeNormalizedFloat,
			bool normalizedCoords = true)
		{
			if (width && height)
			{
				this->~Texture();
				CUDA_CHECK_THROW(cudaMallocArray(&array, &channelDesc, width, height));
				if (src)
				{
					size_t element_size = (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w) / 8;
					CUDA_CHECK_THROW(cudaMemcpy2DToArray(array, 0, 0, src, element_size * width, element_size * width, height, cudaMemcpyHostToDevice));
				}
				cudaResourceDesc resDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = array;
				cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = addressMode;
				texDesc.addressMode[1] = addressMode;
				texDesc.filterMode = filterMode;
				texDesc.readMode = readMode;
				texDesc.normalizedCoords = normalizedCoords;
				CUDA_CHECK_PRINT(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
			}
		}

		operator cudaTextureObject_t()const
		{
			return texture;
		}
	};

	template<>struct Texture<3>
	{
	};

}
#ifdef NotDefinedCUDA_CHECK_THROW
#undef CUDA_CHECK_PRINT
#undef CUDA_CHECK_THROW
#undef FILE_LINE
#undef STR
#undef STRINGIFY
#endif
#undef NotDefinedCUDA_CHECK_THROW