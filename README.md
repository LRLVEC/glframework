# glframework: A light weight multi-window rendering and user interface framework for OpenGL and OpenXR applications

This is an OpenGL and OpenXR based multi-window rendering and user interface system for Windows and Linux desktop, or virtual/augmented/mixed reality. It contains light weight wrappers of OpenGL and OpenXR apis, a multi-window manager and an interface for user-defined imgui UIs.

## Demos

### Galaxy in Hand

<img src="docs/readme/galaxy%20in%20hand.jpg" height = 300/>

GalaxyInHand demo shows how to use OpenXR features, combined with OpenGL backend. It can take user input and create a whole OpenXR instance pipeline. Future work will be focused on 3D ui.

### Multi-simulation

<img src="docs/readme/multi-window%20multi-simulation.png" height = 300/>

MultiSim demo can render multiple n-body simulations in different windows, with **one** global imgui (multi-context imgui is not officially supported yet) interface.

### Multi-view

<img src="docs/readme/multi-window%20single-simulation.png" height = 300/>

MultiView demo can render multiple views of one n-body simulation.

### Multi-thread (To be implemented since imgui multi context is not done yet)

MultiThread demo can render multiple windows with one thread for each window.

### Render texture

<img src="docs/readme/render%20texture%20fractal.png" height = 300/>

RenderTexture demo can render a texture to the window. The Mandelbrot fractal is rendered  to a texture by cuda. This is useful for a CUDA program or off-screen OpenGL program that writes rendering results to a texture.

## Requirements

- CUDA is <span style="color:red">not</span> necessary for building this, but if supported, demos will use it to accelerate.
- A __C++17__ capable compiler. The following choices are recommended and have been tested:
  - __Windows:__ Visual Studio 2019
  - __Linux:__ GCC/G++ 7.5 or higher
- Some libraries that Linux requires is listed in [dependencies.txt](https://github.com/LRLVEC/glframework/blob/master/dependencies.txt).
- To use OpenXR on Windows, please install openxr_loader with vcpkg, and remember to add vcpkg path to cmake configure command.

## Dependencies

- [glew](https://github.com/LRLVEC/glew-cmake/tree/glew-cmake-2.2.0-gitignore): OpenGL requirements.
- [glfw](https://github.com/LRLVEC/glfw/tree/tev): Provide window and context for OpenGL.
- [imgui](https://github.com/LRLVEC/imgui/tree/master): User interface.
- [tiny-stl](https://github.com/LRLVEC/tiny-stl/tree/main): A tiny stl for c++17.

## Compilation

Clone this repository and all its submodules using the following command:

```sh
$ git clone --recursive https://github.com/LRLVEC/glframework.git
$ cd glframework
```

Then, use CMake to build the project: (on Windows, this must be in a [developer command prompt](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_prompt)). If you want to use cuda, add cmake option ```-DGLFRAMEWORK_ENABLE_CUDA=ON```

```sh
glframework$ cmake . -B build -DGLFRAMEWORK_ENABLE_CUDA=OFF
glframework$ cmake --build build --config Release -j
```

## Usage

Add glframework as a dependency of your project:
1. add these to your CMakeLists.txt:

```CMake
get_target_property(GLFRAMEWORK_INCLUDE_DIRECTORIES glframework INCLUDE_DIRECTORIES)
target_include_directories(your_target PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(your_target PUBLIC ${GLFRAMEWORK_LIBRARIES})
```

2. if you don't wish to build demos in your project, add ```set(GLFRAMEWORK_BUILD_DEMOS OFF CACHE BOOL " " FORCE)```
