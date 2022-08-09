#include <cstdio>
#include <_Window.h>
#include <_NBody.h>
#include <_Math.h>
#include <_Time.h>


int main()
{
	/*printf("%d\n", glfwInit());
	Timer timer;
	timer.begin();
	GLFWwindow* window = glfwCreateWindow(500, 500, "ahh", NULL, NULL);
	glfwMakeContextCurrent(window);
	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0.f, 1.f, 0.f, 0.f);
		glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	timer.end();
	timer.print();
	glfwTerminate();*/

	OpenGL::OpenGLInit init(4, 5);
	Window::Window::Data winParameters
	{
		"NBody",
		{
			{800,800},
			true,false
		}
	};
	Window::WindowManager wm(winParameters);
	OpenGL::NBody nBody(5);
	wm.init(0, &nBody);
	init.printRenderer();
	glfwSwapInterval(0);
	FPS fps;
	fps.refresh();
	int i(0);
	while (!wm.close())
	{
		wm.pullEvents();
		wm.render();
		wm.swapBuffers();
		fps.refresh();
		::printf("\r%.2lf    ", fps.fps);
		//fps.printFPS(1);
	}
	return 0;
}