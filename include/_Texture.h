#pragma once
#include <_OpenGL.h>
#include <_File.h>
#include <_BMP.h>

namespace OpenGL
{
	enum TextureStorageType
	{
		TextureStorage1D,
		TextureStorage2D,
		TextureStorage3D,
	};
	enum TextureType
	{
		Texture1D = GL_TEXTURE_1D,
		Texture2D = GL_TEXTURE_2D,
		Texture3D = GL_TEXTURE_3D,
		Texture1DArray = GL_TEXTURE_1D_ARRAY,
		Texture2DArray = GL_TEXTURE_2D_ARRAY,
		TextureRectangle = GL_TEXTURE_RECTANGLE,
		TextureCubeMap = GL_TEXTURE_CUBE_MAP,
		TextureCubeMapArray = GL_TEXTURE_CUBE_MAP_ARRAY,
		TextureTextureBuffer = GL_TEXTURE_BUFFER,
		Texture2DMultiSample = GL_TEXTURE_2D_MULTISAMPLE,
		Texture2DMultiSampleArray = GL_TEXTURE_2D_MULTISAMPLE_ARRAY,
	};
	namespace TextureParameter
	{
		enum Name
		{
			TextureDepthStencilTextureMode = GL_DEPTH_STENCIL_TEXTURE_MODE,
			TextureBaseLevel = GL_TEXTURE_BASE_LEVEL,
			TextureCompareFunc = GL_TEXTURE_COMPARE_FUNC,
			TextureCompareMode = GL_TEXTURE_COMPARE_MODE,
			TextureLodBias = GL_TEXTURE_LOD_BIAS,
			TextureMinFilter = GL_TEXTURE_MIN_FILTER,
			TextureMagFilter = GL_TEXTURE_MAG_FILTER,
			TextureMinLod = GL_TEXTURE_MIN_LOD,
			TextureMaxLod = GL_TEXTURE_MAX_LOD,
			TextureMaxLevel = GL_TEXTURE_MAX_LEVEL,
			TextureSwizzleR = GL_TEXTURE_SWIZZLE_R,
			TextureSwizzleG = GL_TEXTURE_SWIZZLE_G,
			TextureSwizzleB = GL_TEXTURE_SWIZZLE_B,
			TextureSwizzleA = GL_TEXTURE_SWIZZLE_A,
			TextureWarpS = GL_TEXTURE_WRAP_S,
			TextureWarpT = GL_TEXTURE_WRAP_T,
			TextureWarpR = GL_TEXTURE_WRAP_R
		};
		enum Parameter
		{
			DepthStencilTextureMode_DepthComponent = GL_DEPTH_COMPONENT,
			//DepthStencilTextureMode_StencilComponent = GL_STENCIL_COMPONENT,

			CompareFunc_LessEqual = GL_LEQUAL,
			CompareFunc_GreaterEqual = GL_GEQUAL,
			CompareFunc_Less = GL_LESS,
			CompareFunc_Greater = GL_GREATER,
			CompareFunc_Equal = GL_EQUAL,
			CompareFunc_NotEuqal = GL_NOTEQUAL,
			CompareFunc_Always = GL_ALWAYS,
			CompareFunc_Never = GL_NEVER,

			CompareMode_CompareRefToTexture = GL_COMPARE_REF_TO_TEXTURE,
			CompareMode_None = GL_NONE,

			Filter_Nearest = GL_NEAREST,
			Filter_Linear = GL_LINEAR,
			Filter_NearestMipmapNearest = GL_NEAREST_MIPMAP_NEAREST,
			Filter_LinearMipmapNearest = GL_LINEAR_MIPMAP_NEAREST,
			Filter_NearestMipmapLinear = GL_NEAREST_MIPMAP_LINEAR,
			Filter_LinearMipmapLinear = GL_LINEAR_MIPMAP_LINEAR,

			Wrap_ClampToEdge = GL_CLAMP_TO_EDGE,
			Wrap_ClampToBorder = GL_CLAMP_TO_BORDER,
			Wrap_MirroredRepeat = GL_MIRRORED_REPEAT,
			Wrap_Repeat = GL_REPEAT,
			Wrap_MirrorClampToEdge = GL_MIRROR_CLAMP_TO_EDGE,
		};
	}
	enum TextureFormat
	{
		R8 = GL_R8,
		R8Snorm = GL_R8_SNORM,
		R16 = GL_R16,
		R16Snorm = GL_R16_SNORM,
		RG8 = GL_RG8,
		RG8Snorm = GL_RG8_SNORM,
		RG16 = GL_RG16,
		RG16Snorm = GL_RG16_SNORM,
		R3G3B2 = GL_R3_G3_B2,
		RGB4 = GL_RGB4,
		RGB5 = GL_RGB5,
		RGB8 = GL_RGB8,
		RGB8Snorm = GL_RGB8_SNORM,
		RGB10 = GL_RGB10,
		RGB12 = GL_RGB12,
		RGB16Snorm = GL_RGB16_SNORM,
		RGBA2 = GL_RGBA2,
		RGBA4 = GL_RGBA4,
		RGB5A1 = GL_RGB5_A1,
		RGBA8 = GL_RGBA8,
		RGBA8Snorm = GL_RGBA8_SNORM,
		RGB10A2 = GL_RGB10_A2,
		RGB10A2ui = GL_RGB10_A2UI,
		RGBA12 = GL_RGBA12,
		RGBA16 = GL_RGBA16,
		SRGB8 = GL_SRGB8,
		SRGB8Alpha8 = GL_SRGB8_ALPHA8,
		R16f = GL_R16F,
		RG16f = GL_RG16F,
		RGB16f = GL_RGB16F,
		RGBA16f = GL_RGBA16F,
		R32f = GL_R32F,
		RG32f = GL_RG32F,
		RGB32f = GL_RGB32F,
		RGBA32f = GL_RGBA32F,
		R11fG11fB10f = GL_R11F_G11F_B10F,
		RGB9E5 = GL_RGB9_E5,
		R8i = GL_R8I,
		R8ui = GL_R8UI,
		R16i = GL_R16I,
		R16ui = GL_R16UI,
		R32i = GL_R32I,
		R32ui = GL_R32UI,
		RG8i = GL_RG8I,
		RG8ui = GL_RG8UI,
		RG16i = GL_RG16I,
		RG16ui = GL_RG16UI,
		RG32i = GL_RG32I,
		RG32ui = GL_RG32UI,
		RGB8i = GL_RGB8I,
		RGB8ui = GL_RGB8UI,
		RGB16i = GL_RGB16I,
		RGB16ui = GL_RGB16UI,
		RGB32i = GL_RGB32I,
		RGB32ui = GL_RGB32UI,
		RGBA8i = GL_RGBA8I,
		RGBA8ui = GL_RGBA8UI,
		RGBA16i = GL_RGBA16I,
		RGBA16ui = GL_RGBA16UI,
		RGBA32i = GL_RGBA32I,
		RGBA32ui = GL_RGBA32UI,
	};
	enum TextureInputFormat
	{
		TextureInputR = GL_RED,
		TextureInputG = GL_GREEN,
		TextureInputB = GL_BLUE,
		TextureInputRG = GL_RG,
		TextureInputRGB = GL_RGB,
		TextureInputBGR = GL_BGR,
		TextureInputRGBA = GL_RGBA,
		TextureInputBGRA = GL_BGRA,
		TextureInputRInt = GL_RED,
		TextureInputGInt = GL_GREEN,
		TextureInputBInt = GL_BLUE,
		TextureInputRGInt = GL_RG,
		TextureInputRGBInt = GL_RGB,
		TextureInputBGRInt = GL_BGR,
		TextureInputRGBAInt = GL_RGBA,
		TextureInputBGRAInt = GL_BGRA,
		TextureInputDepth = GL_DEPTH_COMPONENT,
		TextureInputStencil = GL_STENCIL_INDEX
	};
	enum TextureInputType
	{
		TextureInputUByte = GL_UNSIGNED_BYTE,
		TextureInputByte = GL_BYTE,
		TextureInputUShort = GL_UNSIGNED_SHORT,
		TextureInputShort = GL_SHORT,
		TextureInputUInt = GL_UNSIGNED_INT,
		TextureInputInt = GL_INT,
		TextureInputFloat = GL_FLOAT,
		TextureInputUByte_3_3_2 = GL_UNSIGNED_BYTE_3_3_2,
		TextureInputUByte_2_3_3_Rev = GL_UNSIGNED_BYTE_2_3_3_REV,
		TextureInputUShort_5_6_5 = GL_UNSIGNED_SHORT_5_6_5,
		TextureInputUShort_5_6_5_Rev = GL_UNSIGNED_SHORT_5_6_5_REV,
		TextureInputUShort_4_4_4_4 = GL_UNSIGNED_SHORT_4_4_4_4,
		TextureInputUShort_4_4_4_4_Rev = GL_UNSIGNED_SHORT_4_4_4_4_REV,
		TextureInputUShort_5_5_5_1 = GL_UNSIGNED_SHORT_5_5_5_1,
		TextureInputUShort_1_5_5_5_Rev = GL_UNSIGNED_SHORT_1_5_5_5_REV,
		TextureInputUInt_8_8_8_8 = GL_UNSIGNED_INT_8_8_8_8,
		TextureInputUInt_8_8_8_8_Rev = GL_UNSIGNED_INT_8_8_8_8_REV,
		TextureInputUInt_10_10_10_2 = GL_UNSIGNED_INT_10_10_10_2,
		TextureInputUInt_2_10_10_10_Rev = GL_UNSIGNED_INT_2_10_10_10_REV
	};
	struct Texture
	{
		struct Data
		{
			virtual void* pointer() = 0;
		};
		Data* data;
		GLuint texture;
		unsigned int binding;
		Texture()
			:
			data(nullptr),
			texture(),
			binding(0)
		{
			create();
		}
		Texture(Data* _data, unsigned int _binding)
			:
			data(_data),
			texture(),
			binding(_binding)
		{
			create();
		}
		void create()
		{
			glGenTextures(1, &texture);
		}
		void bindUnit()
		{
			glBindTextureUnit(binding, texture);
		}
		void del()
		{
			glDeleteTextures(1, &texture);
		}
	};
	struct TextureConfigBase
	{
		Texture* texture;
		TextureType type;
		TextureFormat format;
		unsigned int layers;
		TextureConfigBase()
			:
			texture(nullptr),
			type(Texture2D),
			format(RGB8ui),
			layers(1)
		{
		}
		TextureConfigBase(Texture* _texture, TextureType _type, TextureFormat _format, unsigned int _layers)
			:
			texture(_texture),
			type(_type),
			format(_format),
			layers(_layers)
		{
			bind();
		}
		void bind()
		{
			glBindTexture(type, texture->texture);
		}
		void unBind()
		{
			glBindTexture(type, 0);
		}
		void parameteri(TextureParameter::Name _pname, TextureParameter::Parameter _para)
		{
			glTextureParameteri(texture->texture, _pname, _para);
		}
		void del()
		{
			texture->del();
			unBind();
		}
	};
	template<TextureStorageType>struct TextureConfig :TextureConfigBase
	{
	};
	template<>struct TextureConfig<TextureStorage1D> :TextureConfigBase
	{
		unsigned int width;
		TextureConfig(Texture* _texture, TextureType _type, TextureFormat _format, unsigned int _layers, unsigned int _width)
			:
			TextureConfigBase(_texture, _type, _format, _layers),
			width(_width)
		{
			allocData();
		}
		void allocData()
		{
			glTextureStorage1D(texture->texture, layers, format, width);
		}
		void resize(unsigned int _width)
		{
			width = _width;
			bind();
			glDeleteTextures(1, &texture->texture);
			texture->create();
			bind();
			allocData();
		}
		void dataInit(unsigned int _level, TextureInputFormat _inputFormat, TextureInputType _inputType)
		{
			glTextureSubImage1D(texture->texture, _level, 0, width, _inputFormat, _inputType, texture->data->pointer());
		}
		void dataRefresh(unsigned int _level, TextureInputFormat _inputFormat, TextureInputType _inputType, unsigned int _xOffset, unsigned int _width)
		{
			glTextureSubImage1D(texture->texture, _level, _xOffset, _width, _inputFormat, _inputType, texture->data->pointer());
		}
	};
	template<>struct TextureConfig<TextureStorage2D> :TextureConfigBase
	{
		unsigned int width;
		unsigned int height;
		TextureConfig(Texture* _texture, TextureType _type, TextureFormat _format, unsigned int _layers, unsigned int _width, unsigned int _height)
			:
			TextureConfigBase(_texture, _type, _format, _layers),
			width(_width),
			height(_height)
		{
			allocData();
		}
		void allocData()
		{
			glTextureStorage2D(texture->texture, layers, format, width, height);
		}
		void resize(unsigned int _width, unsigned int _height)
		{
			width = _width;
			height = _height;
			bind();
			glDeleteTextures(1, &texture->texture);
			texture->create();
			bind();
			allocData();
		}
		void dataInit(unsigned int _level, TextureInputFormat _inputFormat, TextureInputType _inputType)
		{
			glTextureSubImage2D(texture->texture, _level, 0, 0, width, height, _inputFormat, _inputType, texture->data ? texture->data->pointer() : nullptr);
		}
		void dataRefresh(unsigned int _level, TextureInputFormat _inputFormat, TextureInputType _inputType, unsigned int _xOffset, unsigned int _yOffset, unsigned int _width, unsigned int _height)
		{
			glTextureSubImage2D(texture->texture, _level, _xOffset, _yOffset, _width, _height, _inputFormat, _inputType, texture->data ? texture->data->pointer() : nullptr);
		}
	};
	template<>struct TextureConfig<TextureStorage3D> :TextureConfigBase
	{
		unsigned int width;
		unsigned int height;
		unsigned int depth;
		TextureConfig(Texture* _texture, TextureType _type, TextureFormat _format, unsigned int _layers, unsigned int _width, unsigned int _height, unsigned int _depth)
			:
			TextureConfigBase(_texture, _type, _format, _layers),
			width(_width),
			height(_height),
			depth(_depth)
		{
			allocData();
		}
		void allocData()
		{
			glTextureStorage3D(texture->texture, layers, format, width, height, depth);
		}
		void resize(unsigned int _width, unsigned int _height, unsigned int _depth)
		{
			width = _width;
			height = _height;
			depth = _depth;
			bind();
			glDeleteTextures(1, &texture->texture);
			texture->create();
			bind();
			allocData();
		}
		void dataInit(unsigned int _level, TextureInputFormat _inputFormat, TextureInputType _inputType)
		{
			glTextureSubImage3D(texture->texture, _level, 0, 0, 0, width, height, depth, _inputFormat, _inputType, texture->data->pointer());
		}
		void dataRefresh(unsigned int _level, TextureInputFormat _inputFormat, TextureInputType _inputType, unsigned int _xOffset, unsigned int _yOffset, unsigned int _zOffset, unsigned int _width, unsigned int _height, unsigned int _depth)
		{
			glTextureSubImage3D(texture->texture, _level, _xOffset, _yOffset, _zOffset, _width, _height, _depth, _inputFormat, _inputType, texture->data->pointer());
		}
	};
	struct TextureCube
	{
		struct Data
		{
			virtual void* pointer(unsigned int) = 0;
		};
		Data* data;
		GLuint texture;
		unsigned int binding;
		TextureFormat format;
		unsigned int layers;
		unsigned int width;
		unsigned int height;
		TextureCube(Data* _data, unsigned int _binding, TextureFormat _format, unsigned int _layers, unsigned int _width, unsigned int _height)
			:
			data(_data),
			texture(),
			binding(_binding),
			format(_format),
			layers(_layers),
			width(_width),
			height(_height)
		{
			create();
			bind();
			allocData();
		}
		void create()
		{
			glGenTextures(1, &texture);
		}
		void bindUnit()
		{
			glBindTextureUnit(binding, texture);
		}
		void bind()
		{
			glBindTexture(TextureCubeMap, texture);
		}
		void unBind()
		{
			glBindTexture(TextureCubeMap, 0);
		}
		void parameteri(TextureParameter::Name _pname, TextureParameter::Parameter _para)
		{
			glTextureParameteri(texture, _pname, _para);
		}
		void allocData()
		{
			glTextureStorage2D(texture, layers, format, width, height);
		}
		void dataInit(unsigned int _level, TextureInputFormat _inputFormat, TextureInputType _inputType)
		{
			for (int c0(0); c0 < 6; ++c0)
				glTextureSubImage3D(texture, _level, 0, 0, c0, width, height, 1, _inputFormat, _inputType, data->pointer(c0));
		}
	};
	struct BMPData :Texture::Data
	{
		BMP bmp;
		BMPData(String<char>const& _path)
			:
			bmp(_path)
		{
		}
		virtual void* pointer()
		{
			return bmp.data_24;
		}
	};
	struct BMPCubeData :TextureCube::Data
	{
		BMP bmp[6];
		BMPCubeData(String<char>const& _path)
			:
			bmp{ _path + "right.bmp",_path + "left.bmp",_path + "front.bmp",_path + "back.bmp",_path + "up.bmp",_path + "down.bmp" }
		{
		}
		virtual void* pointer(unsigned int n)override
		{
			return bmp[n].data_24;
		}
	};

	struct FrameBuffer;
	struct FrameBufferSource
	{
		FrameBufferAttachment attachment;
		FrameBufferSource(FrameBufferAttachment _attachment)
			:
			attachment(_attachment)
		{
		}
		virtual void bind(FrameBuffer* _frameBuffer, uint32_t _layer = 0) = 0;
		virtual void set_texture(GLuint _texture) {}
		virtual bool is_texture()const = 0;
	};
	struct FrameBuffer
	{
		GLuint frameBuffer;
		uint32_t width;
		uint32_t height;
		bool resized;
		FrameBufferSource* color;
		FrameBufferSource* depth;

		FrameBuffer(bool _create = false)
			:
			frameBuffer(0),
			width(0),
			height(0),
			resized(true),
			color(nullptr),
			depth(nullptr)
		{
			if (_create)create();
		}

		~FrameBuffer()
		{
			unbind();
			color = nullptr;
			depth = nullptr;
			glDeleteFramebuffers(1, &frameBuffer);
		}

		void create()
		{
			glGenFramebuffers(1, &frameBuffer);
		}

		void bind(GLuint color_texture = 0, GLuint depth_texture = 0, uint32_t color_layer = 0, uint32_t depth_layer = 0)
		{
			glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

			// make sure that we can get texture size and then pass to render buffer to resize
			if(color->is_texture())
			{
				color->set_texture(color_texture);
				color->bind(this, color_layer);
				depth->set_texture(depth_texture);
				depth->bind(this, depth_layer);
			}
			else
			{
				depth->set_texture(depth_texture);
				depth->bind(this, depth_layer);
				color->set_texture(color_texture);
				color->bind(this, color_layer);
			}
		}

		void unbind()
		{
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
	};
	struct FrameBufferSourceTexture : FrameBufferSource
	{
		GLuint texture;
		TextureType type;

		FrameBufferSourceTexture(FrameBufferAttachment _attachment, TextureType _type)
			:
			FrameBufferSource(_attachment),
			texture(0),
			type(_type)
		{
		}

		void bind(FrameBuffer* _frameBuffer, uint32_t _layer = 0)
		{
			if (texture)
			{
				GLint _width(0), _height(0);
				glBindTexture(type, texture);
				glGetTexLevelParameteriv(type, 0, GL_TEXTURE_WIDTH, &_width);
				glGetTexLevelParameteriv(type, 0, GL_TEXTURE_HEIGHT, &_height);
				if (_frameBuffer->width != _width || _frameBuffer->height != _height)
				{
					_frameBuffer->resized = true;
					_frameBuffer->width = _width;
					_frameBuffer->height = _height;
				}
				switch (type)
				{
				case Texture2D:
					glFramebufferTexture2D(GL_FRAMEBUFFER, GLenum(attachment), Texture2D, texture, 0);
					break;
				case Texture3D:
					glFramebufferTexture3D(GL_FRAMEBUFFER, GLenum(attachment), Texture3D, texture, 0, _layer);
					break;
				default:break;
				}
			}
		}

		void set_texture(GLuint _texture)
		{
			texture = _texture;
		}

		bool is_texture()const
		{
			return true;
		}
	};
	struct FrameBufferSourceRenderBuffer : FrameBufferSource
	{
		RenderBuffer renderBuffer;

		FrameBufferSourceRenderBuffer(FrameBufferAttachment _attachment, RenderBufferFormat _format)
			:
			FrameBufferSource(_attachment),
			renderBuffer(_format)
		{
		}

		void bind(FrameBuffer* _frameBuffer, uint32_t _layer = 0)
		{
			if (_frameBuffer->resized)
				renderBuffer.allocData(_frameBuffer->width, _frameBuffer->height);
			renderBuffer.attach(attachment);
		}
		
		bool is_texture()const
		{
			return false;
		}
	};
}
