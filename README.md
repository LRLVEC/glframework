# glframework: A light weight multi-window rendering and user interface framework for OpenGL applications

<img src="docs/readme/multi-window%20multi-simulation.png"/>

This is a OpenGL based multi-window rendering and user interface system for Windows and Linux desktop. As the test demo shows, it can render multiple n-body simulations in different windows, with **one** global imgui (multi-context imgui is not officially supported yet) interface. It contains a light weight wrapper of OpenGL api, a multi-window manager and an interface for user-defined imgui UIs.
# Requirements
- CUDA is <span style="color:red">not</span> necessary for building this, but if supported, test demo will use it to accelerate.
- A __C++17__ capable compiler. The following choices are recommended and have been tested:
  - __Windows:__ Visual Studio 2019
  - __Linux:__ GCC/G++ 7.5 or higher
# Dependencies
- [glew](https://github.com/LRLVEC/glew-cmake/tree/glew-cmake-2.2.0-gitignore): OpenGL requirements.
- [glfw](https://github.com/LRLVEC/glfw/tree/tev): Provide window and context for OpenGL.
- [imgui](https://github.com/LRLVEC/imgui/tree/master): User interface.
- [tiny-stl](https://github.com/LRLVEC/tiny-stl/tree/main): A tiny stl for c++17.

# Compilation
Clone this repository and all its submodules using the following command:
```sh
$ git clone --recursive https://github.com/LRLVEC/glframework.git
$ cd glframework
```

Then, use CMake to build the project: (on Windows, this must be in a [developer command prompt](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_prompt)). If you want to use cuda, add cmake option ```-DGLFRAMEWORK_ENABLE_CUDA=ON```
```sh
glframework$ cmake . -B build -DGLFRAMEWORK_ENABLE_CUDA=OFF
glframework$ cmake --build build --config RelWithDebInfo -j
```

# Usage

Add glframework as a dependency of your project:
1. add these to your CMakeLists.txt:
```
get_target_property(GLFRAMEWORK_INCLUDE_DIRECTORIES glframework INCLUDE_DIRECTORIES)
target_include_directories(your_target PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(your_target PUBLIC ${GLFRAMEWORK_LIBRARIES})
```
1. if you don't wish to build tests in your project, add ```set(GLFRAMEWORK_BUILD_TESTS OFF CACHE BOOL " " FORCE)```