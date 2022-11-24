#pragma once
#include <_Texture.h>
#include <_Math.h>

namespace OpenGL
{
	struct TextureRendererProgram : Program
	{
		struct TriangleData : Buffer::Data
		{
			using Vertex = Math::vec2<float>;
			using Triangle = Array<Vertex, 3>;
			Array<Triangle, 2> triangles;
			TriangleData()
				:
				Data(StaticDraw),
				triangles({ {{-1,-1},{1,-1},{1,1}},{{1,1},{-1,1},{-1,-1}} })
			{
			}
			virtual void* pointer()override
			{
				return (void*)triangles.data;
			}
			virtual unsigned int size()override
			{
				return sizeof(Triangle) * triangles.length;
			}
		};
		
		Texture frameTexture;
		TextureConfig<TextureStorage2D> frameConfig;
		TriangleData triangles;
		Buffer trianglesBuffer;
		BufferConfig bufferArray;
		VertexAttrib positions;
		bool updated;

		TextureRendererProgram(SourceManager* _sourceManager, FrameScale const& _size)
			:
			Program(_sourceManager, "TextureRenderer", Vector<VertexAttrib*>{&positions}),
			frameTexture(nullptr, 0),
			frameConfig(&frameTexture, Texture2D, RGBA32f, 1, _size.w, _size.h),
			triangles(),
			trianglesBuffer(&triangles),
			bufferArray(&trianglesBuffer, ArrayBuffer),
			positions(&bufferArray, 0, VertexAttrib::two, VertexAttrib::Float, false, sizeof(TriangleData::Vertex), 0, 0),
			updated(false)
		{
			init();
			bufferArray.dataInit();
			use();
			frameTexture.bindUnit();
		}
		GLuint getRenderTexture()const
		{
			return frameTexture.texture;
		}
		FrameScale size()const
		{
			return{ int(frameConfig.width), int(frameConfig.height) };
		}
		virtual void initBufferData()override
		{
		}
		virtual void run() override
		{
			if (updated)
			{
				// pixelPixel.bind();
				// frameConfig.dataInit(0, TextureInputRGBA, TextureInputFloat);
				updated = false;
				// pixelPixel.unbind();
			}
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			glDrawArrays(GL_TRIANGLES, 0, 6);
		}
		void resize(FrameScale const& _size)
		{
			glViewport(0, 0, _size.w, _size.h);
			frameConfig.resize(_size.w, _size.h);
			// pixelPixel.dataInit();
		}
	};
}