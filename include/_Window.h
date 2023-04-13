#pragma once
#include <_OpenGL.h>

#include <_String.h>
#include <_List.h>
#include <stdexcept>
#include <memory>

namespace Window
{
	struct Window
	{
		struct CallbackFun
		{
			struct Frame
			{
				GLFWframebuffersizefun size;
				GLFWwindowposfun pos;
				GLFWwindowfocusfun focus;
			};
			struct Input
			{
				GLFWmousebuttonfun button;
				GLFWcursorposfun pos;
				GLFWscrollfun scroll;
				GLFWkeyfun key;
			};
			Frame frame;
			Input input;
		};
		struct Data
		{
			struct Size
			{
				OpenGL::FrameScale size;
				bool resizable;
				bool fullScreen;
			};
			char const* title;
			Size size;
		};

		struct Title
		{
			String<char> title;
			Title() = default;
			Title(char const* _title)
				:
				title(_title)
			{
			}
			bool operator==(char const* a)const
			{
				return title == a;
			}
			bool operator==(String<char> const& a)const
			{
				return title == a;
			}
			void init(GLFWwindow* _window)
			{
				glfwSetWindowTitle(_window, title.data);
			}
		};
		struct Monitor
		{
			GLFWmonitor* monitor;
			GLFWmonitor** monitors;
			GLFWvidmode mode;
			String<char>name;
			int num;
			int numAll;

			Monitor()
				:
				monitor(nullptr),
				monitors(nullptr),
				mode(),
				name(),
				num(-1),
				numAll(0)
			{
			}
			void init()
			{
				monitors = glfwGetMonitors(&numAll);
			}
			bool search(int _w, int _h)
			{
				if (!monitors)init();
				for (int c0 = 0; c0 < numAll; ++c0)
				{
					GLFWvidmode const* _mode(glfwGetVideoMode(monitors[c0]));
					if (_mode->width == _w && _mode->height == _h)
					{
						mode = *_mode;
						monitor = monitors[c0];
						num = c0;
						return true;
					}
				}
				mode = *glfwGetVideoMode(monitors[0]);
				monitor = monitors[0];
				num = 0;
				return false;
			}
			String<char>& getName()
			{
				if (monitor)
					return name = glfwGetMonitorName(monitor);
				else
					return name = "No monitor!!";
			}
		};
		struct Size
		{
			struct FullScreen
			{
				bool fullScreen;
				FullScreen()
					:
					fullScreen(false)
				{

				}
				FullScreen(bool _fullScreen)
					:
					fullScreen(_fullScreen)
				{

				}
			};

			bool resizable;
			OpenGL::FrameScale size;
			FullScreen fullScreen;
			Size(OpenGL::FrameScale _size)
				:
				size(_size),
				resizable(true),
				fullScreen(false)
			{
				glfwWindowHint(GLFW_RESIZABLE, true);
			}
			Size(OpenGL::FrameScale _size, bool _resizable, bool _fullScreen)
				:
				size(_size),
				resizable(_resizable),
				fullScreen(_fullScreen)
			{
				glfwWindowHint(GLFW_RESIZABLE, _resizable);
			}
			void set(GLFWwindow* _window, int _w, int _h)
			{
				if (!fullScreen.fullScreen && resizable)
					glfwSetWindowSize(_window, size.w = _w, size.h = _h);
			}
		};
		struct Callback
		{
			struct Frame
			{
				Window::CallbackFun::Frame frame;
				Frame() = delete;
				Frame(Window::CallbackFun::Frame const& _input)
					:
					frame(_input)
				{
				}
				void init(GLFWwindow* _window)
				{
					glfwSetFramebufferSizeCallback(_window, frame.size);
					glfwSetWindowPosCallback(_window, frame.pos);
					glfwSetWindowFocusCallback(_window, frame.focus);
				}
			};
			struct Input
			{
				Window::CallbackFun::Input input;
				Input() = delete;
				Input(Window::CallbackFun::Input const& _input)
					:
					input(_input)
				{
				}
				void init(GLFWwindow* _window)
				{
					glfwSetMouseButtonCallback(_window, input.button);
					glfwSetCursorPosCallback(_window, input.pos);
					glfwSetScrollCallback(_window, input.scroll);
					glfwSetKeyCallback(_window, input.key);
				}
			};
			Frame frame;
			Input input;
			Callback() = delete;
			Callback(
				Window::CallbackFun::Frame const& _frameIn,
				Window::CallbackFun::Input const& _iinputIn)
				:
				frame(_frameIn),
				input(_iinputIn)
			{
			}
			void init(GLFWwindow* _window)
			{
				frame.init(_window);
				input.init(_window);
			}
		};

		static bool glewInitialized;

		GLFWwindow* window;
		Title title;
		Monitor monitor;
		Size size;
		Callback callback;
		OpenGL::OpenGL* openGL;

		Window() = delete;
		Window(Window const&) = default;
		Window(Data const& _data, CallbackFun const& _callback, bool _decorated, bool _show)
			:
			window(nullptr),
			title(_data.title),
			monitor(),
			size(_data.size.size, _data.size.resizable, _data.size.fullScreen),
			callback(_callback.frame, _callback.input),
			openGL(nullptr)
		{
			OpenGL::OpenGLInit::setWindowOpenGLVersion();
			glfwWindowHint(GLFW_DECORATED, _decorated);
			glfwWindowHint(GLFW_VISIBLE, _show);
			// glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_FLUSH);
			glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_NONE);
			if (size.fullScreen.fullScreen)
			{
				monitor.search(size.size.w, size.size.h);
				window = glfwCreateWindow(
					size.size.w = monitor.mode.width,
					size.size.h = monitor.mode.height,
					title.title, monitor.monitor,
					NULL);
			}
			else
				window = glfwCreateWindow(size.size.w, size.size.h, title.title, NULL, NULL);
			glfwMakeContextCurrent(window);
			if (!glewInitialized)
			{
				glewInitialized = true;
				glewInit();
			}
			//callback.init(window);
			//openGL->init(size.size);
		}
		Window(Data const& _data, CallbackFun const& _callback)
			:Window(_data, _callback, true, true)
		{
		}
		~Window()
		{
			if (window)
			{
				if (glfwWindowShouldClose(window))
				{
					printf("Destroy window %p\n", window);
					glfwDestroyWindow(window);
				}
				window = nullptr;
			}
		}
		bool operator==(Window const& _window)const
		{
			return window == _window.window;
		}
		bool operator==(GLFWwindow* const _window) const
		{
			return window == _window;
		}
		void init(OpenGL::OpenGL* _openGL)
		{
			makeCurrent();
			openGL = _openGL;
			callback.init(window);
			openGL->init(size.size);
		}
		void setTitle(char const* _title)
		{
			glfwSetWindowTitle(window, _title);
		}
		void makeCurrent()const
		{
			glfwMakeContextCurrent(window);
		}
		void swapBuffers()const
		{
			glfwSwapBuffers(window);
		}
	};

	// Window Manager without imgui
	struct WindowManager
	{
		static WindowManager* __windowManager;
		static void frameSizeCallback(GLFWwindow* _window, int _w, int _h)
		{
			glfwMakeContextCurrent(_window);
			__windowManager->find(_window).openGL->frameSize(_window, _w, _h);
		}
		static void framePosCallback(GLFWwindow* _window, int _w, int _h)
		{
			glfwMakeContextCurrent(_window);
			WindowManager::__windowManager->find(_window).openGL->framePos(_window, _w, _h);
		}
		static void frameFocusCallback(GLFWwindow* _window, int _focused)
		{
			glfwMakeContextCurrent(_window);
			WindowManager::__windowManager->find(_window).openGL->frameFocus(_window, _focused);
		}
		static void mouseButtonCallback(GLFWwindow* _window, int _button, int _action, int _mods)
		{
			glfwMakeContextCurrent(_window);
			WindowManager::__windowManager->find(_window).openGL->mouseButton(_window, _button, _action, _mods);
		}
		static void mousePosCallback(GLFWwindow* _window, double _x, double _y)
		{
			glfwMakeContextCurrent(_window);
			WindowManager::__windowManager->find(_window).openGL->mousePos(_window, _x, _y);
		}
		static void mouseScrollCallback(GLFWwindow* _window, double _x, double _y)
		{
			glfwMakeContextCurrent(_window);
			WindowManager::__windowManager->find(_window).openGL->mouseScroll(_window, _x, _y);
		}
		static void keyCallback(GLFWwindow* _window, int _key, int _scancode, int _action, int _mods)
		{
			glfwMakeContextCurrent(_window);
			WindowManager::__windowManager->find(_window).openGL->key(_window, _key, _scancode, _action, _mods);
		}
		static constexpr Window::CallbackFun callbackFun
		{
			{
				frameSizeCallback,
				framePosCallback,
				frameFocusCallback
			},
			{
				mouseButtonCallback,
				mousePosCallback,
				mouseScrollCallback,
				keyCallback
			}
		};

		List<Window> windows;

		WindowManager()
		{
			if (__windowManager)throw std::runtime_error{ "Cannot construct new WindowManager if there exists one!" };
			__windowManager = this;
		}
		WindowManager(Window::Data const& _data) : WindowManager()
		{
			windows.pushBack(Window(_data, callbackFun));
		}
		void init(unsigned int _num, OpenGL::OpenGL* _openGL)
		{
			Window& window = windows[_num].data;
			window.init(_openGL);
		}
		void createWindow(Window::Data const& _data)
		{
			windows.pushBack(Window(_data, callbackFun));
		}
		void makeCurrent(unsigned int _num)
		{
			windows[_num].data.makeCurrent();
		}
		Window& find(GLFWwindow* const _window)
		{
			return windows.find(_window).data;
		}
		bool exists(Window* _window)const
		{
			return windows.id(*_window) >= 0;
		}
		void render()
		{
			windows.traverse
			([](Window const& _window)
				{
					glfwMakeContextCurrent(_window.window);
					if (_window.openGL)
					{
						_window.openGL->run();
					}
					return true;
				}
			);
		}
		void swapBuffers()
		{
			windows.traverse
			([](Window const& _window)
				{
					_window.makeCurrent();
					_window.swapBuffers();
					return true;
				}
			);
		}
		void pullEvents()
		{
			glfwPollEvents();
		}
		bool close()
		{
			windows.check([](Window const& _window)
				{
					if (glfwWindowShouldClose(_window.window))
					{
						glfwMakeContextCurrent(_window.window);
						glfwSetWindowShouldClose(_window.window, true);
						_window.openGL->close();
						//glfwDestroyWindow(_window.window);// move to deconstruction func
						return false;
					}
					return true;
				});
			if (!windows.length)return true;
			return false;
		}
		void closeAll()
		{
			windows.check([](Window const& _window)
				{
					glfwMakeContextCurrent(_window.window);
					glfwSetWindowShouldClose(_window.window, true);
					_window.openGL->close();
					return false;
				}
			);
		}
	};
}
