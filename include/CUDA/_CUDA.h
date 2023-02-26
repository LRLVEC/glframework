#pragma once
#include <_Texture.h>
// #include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <time.h>
#include <random>
#include <_BMP.h>
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
	struct Buffer
	{
		enum BufferType
		{
			Device,
			GLinterop,
			ZeroCopy,
			Unused,
		};
		BufferType type;
		size_t size;
		size_t hostSize;
		cudaGraphicsResource* graphics;
		GLuint gl;
		void* device;
		void* host;

		Buffer(BufferType _type)
			:
			type(_type),
			size(0),
			hostSize(0),
			graphics(nullptr),
			gl(0),
			device(nullptr),
			host(nullptr)
		{
		}
		Buffer(BufferType _type, unsigned long long _size)
			:
			Buffer(_type)
		{
			size = _size;
			switch (_type)
			{
			case Device:
			case ZeroCopy:
			{
				resize(size_t(_size));
				break;
			}
			case GLinterop:
			{
				resize(GLuint(_size));
				break;
			}
			}
		}
		Buffer(GLuint _gl)
			:
			type(GLinterop),
			size(0),
			hostSize(0),
			graphics(nullptr),
			device(nullptr),
			host(nullptr)
		{
			resize(_gl);
		}
		template<class T>Buffer(T const& a, bool copy)
			:
			type(Device),
			size(0),
			hostSize(0),
			graphics(nullptr),
			device(nullptr),
			host(nullptr)
		{
			resize(sizeof(T));
			if (copy)CUDA_CHECK_THROW(cudaMemcpy(device, &a, size, cudaMemcpyHostToDevice));
		}
		~Buffer()
		{
			if (type != Unused)
				switch (type)
				{
				case Device:
				{
					freeHost();
					CUDA_CHECK_PRINT(cudaFree(device));
					break;
				}
				case GLinterop:
				{
					unmap();
					freeHost();
					break;
				}
				case ZeroCopy:
				{
					CUDA_CHECK_PRINT(cudaFreeHost(host));
					break;
				}
				}
			type = Unused;
			size = 0;
			hostSize = 0;
			graphics = nullptr;
			gl = 0;
			host = nullptr;
			device = nullptr;
		}
		void printInfo(char const* a)const
		{
			::printf("%s", a);
			::printf("[Type: ");
			switch (type)
			{
			case Device: ::printf("Device"); break;
			case GLinterop: ::printf("GLinterop"); break;
			case ZeroCopy: ::printf("ZeroCopy"); break;
			case Unused: ::printf("Unused"); break;
			}
			::printf(", Size: %llu, HostSize: %llu, GR: 0x%p, GL: %u, Device: 0x%p, Host: 0x%p]\n",
				size, hostSize, graphics, gl, device, host);
		}
		//Doesn't keep previous data...
		void resize(size_t _size)
		{
			size = _size;
			switch (type)
			{
			case Device:
			{
				CUDA_CHECK_THROW(cudaFree(device));
				CUDA_CHECK_THROW(cudaMalloc(&device, _size));
				break;
			}
			case ZeroCopy:
			{
				CUDA_CHECK_THROW(cudaFreeHost(host));
				CUDA_CHECK_THROW(cudaHostAlloc(&host, _size, cudaHostAllocPortable | cudaHostAllocMapped));
				CUDA_CHECK_THROW(cudaHostGetDevicePointer(&device, host, 0));
				break;
			}
			case GLinterop:break;
			}
		}
		void resize(GLuint _gl)
		{
			//bug here!!!!!
			CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&graphics, gl = _gl, cudaGraphicsRegisterFlagsNone));
			//map();
			//unmap();
		}
		void resizeHost()
		{
			if (size != hostSize)::free(host);
			if (size)host = ::malloc(hostSize = size);
		}
		void* map()
		{
			if (type == GLinterop)
			{
				CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &graphics));
				CUDA_CHECK_THROW(cudaGraphicsResourceGetMappedPointer(&device, &size, graphics));
			}
			return device;
		}
		void unmap()
		{
			if (type == GLinterop)
			{
				if (graphics)
				{
					CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &graphics));
					graphics = nullptr;
				}
				device = nullptr;
			}
			else CUDA_CHECK_THROW(cudaStreamSynchronize(0));
		}
		void freeHost()
		{
			::free(host);
			host = nullptr;
			hostSize = 0;
		}
		void moveToHost()
		{
			if (host && device)
			{
				cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
			}
		}
		void moveToDevice()
		{
			if (type == Device && size && hostSize == size)
			{
				cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
			}
		}
		template<class T>void copy(T const& a)
		{
			if (size == 0 && type != GLinterop)resize(sizeof(T));
			if (size >= sizeof(T))
				CUDA_CHECK_THROW(cudaMemcpy(device, &a, sizeof(T), cudaMemcpyHostToDevice));
		}
		void copy(void* _src, size_t _size)
		{
			if (size == 0 && type != GLinterop)resize(_size);
			if (_size)
			{
				if (size >= _size)CUDA_CHECK_THROW(cudaMemcpy(device, _src, _size, cudaMemcpyHostToDevice));
			}
			else CUDA_CHECK_THROW(cudaMemcpy(device, _src, size, cudaMemcpyHostToDevice));
		}
		void copy(Buffer& a)
		{
			type = a.type;
			size = a.size;
			graphics = a.graphics;
			gl = a.gl;
			device = a.device;
			host = a.host;
			a.type = Unused;
		}
		void clearDevice(int val)
		{
			if (device)CUDA_CHECK_THROW(cudaMemset(device, val, size));
		}
		// deprecated in cuda 12.0
		// operator CUdeviceptr()const
		// {
		// 	return (CUdeviceptr)device;
		// }
	};
	// mapping OpenGL Texture, can read and write
	struct GLTextureBase
	{
		cudaGraphicsResource_t graphicsResources;

		GLTextureBase():graphicsResources(nullptr) {}

		void unregister_resource()
		{
			if (graphicsResources)
			{
				CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(graphicsResources));
				graphicsResources = nullptr;
			}
		}

		void map(cudaStream_t _stream)
		{
			CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &graphicsResources, _stream));
		}

		void unmap(cudaStream_t _stream)
		{
			CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &graphicsResources, _stream));
		}
	};
	template<unsigned int dim>struct GLTexture:GLTextureBase
	{
		static_assert(dim&& dim < 4, "Dim must be one of 1, 2, 3!");
	};
	template<>struct GLTexture<1>:GLTextureBase
	{
	};
	template<>struct GLTexture<2>:GLTextureBase
	{
		cudaArray_t array;				// read only
		cudaSurfaceObject_t surface;	// read and write
		cudaTextureObject_t texture;	// read and write

		GLTexture():array(nullptr), surface(0) {}

		void registerImage(OpenGL::TextureConfig<OpenGL::TextureStorage2D> const& _textureConfig, unsigned int _flags)
		{
			if (graphicsResources)
			{
				unregister_resource();
			}
			CUDA_CHECK_THROW(cudaGraphicsGLRegisterImage(&graphicsResources, _textureConfig.texture->texture, _textureConfig.type, _flags));
		}

		cudaArray_t createArray(unsigned int _level)
		{
			CUDA_CHECK_THROW(cudaGraphicsSubResourceGetMappedArray(&array, graphicsResources, 0, _level));
			return array;
		}

		cudaSurfaceObject_t createSurface()
		{
			cudaResourceDesc resourceDesc;
			memset(&resourceDesc, 0, sizeof(resourceDesc));
			resourceDesc.resType = cudaResourceTypeArray;
			resourceDesc.res.array.array = array;
			CUDA_CHECK_THROW(cudaCreateSurfaceObject(&surface, &resourceDesc));
			return surface;
		}

		cudaTextureObject_t createTexture(
			cudaTextureFilterMode cudaFilterMode = cudaFilterModeLinear,
			cudaTextureAddressMode cudaAddressMode = cudaAddressModeClamp
		)
		{
			cudaResourceDesc resourceDesc;
			memset(&resourceDesc, 0, sizeof(resourceDesc));
			resourceDesc.resType = cudaResourceTypeArray;
			resourceDesc.res.array.array = array;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.filterMode = cudaFilterMode;
			texDesc.normalizedCoords = true;
			texDesc.addressMode[0] = cudaAddressMode;
			texDesc.addressMode[1] = cudaAddressMode;
			texDesc.addressMode[2] = cudaAddressMode;

			CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resourceDesc, &texDesc, nullptr));
			return texture;
		}

		void destroySurface()
		{
			CUDA_CHECK_THROW(cudaDestroySurfaceObject(surface));
			surface = 0;
		}

		void destroyTexture()
		{
			CUDA_CHECK_THROW(cudaDestroyTextureObject(texture));
			texture = 0;
		}
	};
	template<>struct GLTexture<3>:GLTextureBase
	{
	};
	struct CubeMap
	{
		BMP::Pixel_32* data;
		unsigned int width;

		CubeMap():data(nullptr), width(0) {}
		CubeMap(String<char>const& _path):data(nullptr), width(0)
		{
			String<char> names[6]{ "right.bmp","left.bmp" ,"top.bmp" ,"bottom.bmp"  ,"back.bmp","front.bmp" };
			BMP tp(_path + names[0], true);
			width = tp.header.width;
			size_t sz(sizeof(BMP::Pixel_32) * width * width);
			data = (BMP::Pixel_32*)malloc(6 * sz);
			memcpy(data, tp.data_32, sz);
			for (int c0(1); c0 < 6; ++c0)
			{
				BMP ts(_path + names[c0], true);
				memcpy(data + c0 * sz / 4, ts.data_32, sz);
			}
		}
		~CubeMap()
		{
			::free(data);
			data = nullptr;
		}
		void moveToGPU(cudaArray* _cuArray)const
		{
			cudaMemcpy3DParms cpy3Dparams
			{
				nullptr,{0,0,0},{data, width * 4ll,width, width},
				_cuArray,{0,0,0},{0},{ width, width, 6 }, cudaMemcpyHostToDevice
			};
			CUDA_CHECK_THROW(cudaMemcpy3D(&cpy3Dparams));
		}
	};

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

		Texture(size_t width, size_t height,
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
			if (width && height)
			{
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

	template<>struct Texture<3>
	{
	};

	struct TextureCube
	{
		cudaArray* data;
		cudaTextureObject_t textureObj;

		TextureCube(cudaChannelFormatDesc const& _cd, cudaTextureFilterMode _fm,
			cudaTextureReadMode _rm, bool normalizedCoords, CubeMap const& cubeMap)
			:
			data(nullptr),
			textureObj(0)
		{
			if (cubeMap.width)
			{
				cudaExtent extent{ cubeMap.width, cubeMap.width, 6 };
				CUDA_CHECK_THROW(cudaMalloc3DArray(&data, &_cd, extent, cudaArrayCubemap));
				cubeMap.moveToGPU(data);
				cudaResourceDesc resDesc;
				cudaTextureDesc texDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				memset(&texDesc, 0, sizeof(texDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = data;
				texDesc.normalizedCoords = normalizedCoords;
				texDesc.filterMode = _fm;
				texDesc.addressMode[0] = cudaAddressModeWrap;
				texDesc.addressMode[1] = cudaAddressModeWrap;
				texDesc.addressMode[2] = cudaAddressModeWrap;
				texDesc.readMode = _rm;
				CUDA_CHECK_PRINT(cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr));
			}
		}
		~TextureCube()
		{
			if (textureObj)
			{
				CUDA_CHECK_PRINT(cudaDestroyTextureObject(textureObj));
				textureObj = 0;
			}
			if (data)
			{
				cudaFreeArray(data);
				data = nullptr;
			}
		}
		operator cudaTextureObject_t()const
		{
			return textureObj;
		}
	};

	struct OpenGLDeviceInfo
	{
		unsigned int deviceCount;
		int devices[8];
		OpenGLDeviceInfo()
			:
			devices{ -1,-1,-1,-1,-1,-1,-1,-1 }
		{
			CUDA_CHECK_THROW(cudaGLGetDevices(&deviceCount, devices, 8, cudaGLDeviceListAll));
		}
		void printInfo()
		{
			::printf("Number of CUDA devices corresponding to the current OpenGL context:\t%u\n", deviceCount);
			::printf("Devices:\t");
			for (int c0(0); c0 < deviceCount; ++c0)
				::printf("%d\t", devices[c0]);
			::printf("\n");
		}
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