#pragma once
#include <xr/_OpenXR.h>

namespace XrOS
{
	// standard OpenXR app interface, relies on XrOS
	struct XrApp
	{
		virtual void init(XrVector3f const& size) {}
		virtual void run() {}

		// virtual void pollEvent(XrEventDataBaseHeader const* event) {}//should be placed in session manager
		// session manager should cooperate with imgui window manager
		// maybe we should have sth like volume for each session? just like multi-window system, we have
		// multi volume system.

		virtual void sessionStateChange() {}
	};

	struct VolumeCreateInfo
	{
		char const* title;
		bool fullScreen;
		XrPosef pose;
		XrVector3f size;
	};

	struct Volume
	{
		std::string title;
		bool fullScreen;
		XrPosef pose; // world to volume
		XrVector3f size;
		XrVector3f speed;
		XrVector3f acceleration;
		OpenGL::FrameBuffer frameBuffer;
		OpenGL::FrameBufferSourceTexture color;
		OpenGL::FrameBufferSourceTexture depth;
		XrApp* xrApp;

		Volume() = delete;
		Volume(VolumeCreateInfo const& _createInfo)
			:
			title(_createInfo.title),
			fullScreen(_createInfo.fullScreen),
			pose(_createInfo.pose),
			size(_createInfo.size),
			speed{ 0 },
			acceleration{ 0 },
			frameBuffer(false),
			color(OpenGL::FrameBufferAttachment::Color0, OpenGL::Texture3D),
			depth(OpenGL::FrameBufferAttachment::Depth, OpenGL::Texture3D),
			xrApp(nullptr)
		{

		}

		void init(XrApp* _xrApp)
		{
			xrApp = _xrApp;
			// todo: bind actions
			xrApp->init(size);
		}
	};
}