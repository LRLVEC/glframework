cmake_minimum_required(VERSION 3.18)
project(glframework LANGUAGES CXX)

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/CMakeLists.txt")
	message(FATAL_ERROR
		"Some dependencies are missing. "
		"If you forgot the \"--recursive\" flag when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()

option(GLFRAMEWORK_BUILD_DEMOS "Build glframework test programs" ON)
option(GLFRAMEWORK_ENABLE_CUDA "Build glframework with cuda implemented NBody" ON)
option(GLFRAMEWORK_BUILD_OPENXR "Build glframework with OpenXR support" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
set(GLFW_VULKAN_STATIC OFF CACHE BOOL " " FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

# Add GLFW
add_subdirectory(dependencies/glfw)

set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
mark_as_advanced(
	GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
	GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
	GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
	BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
)

list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/include")

set(TINY_STL_BUILD_TESTS OFF CACHE BOOL " " FORCE)
add_subdirectory(dependencies/tiny-stl)
mark_as_advanced(TINY_STL_BUILD_TESTS)
list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/tiny-stl/include")
list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui")
list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies")
list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Add GLEW
if (MSVC)
# Build glew from local repo
	message("Use MSVC!")
	set(ONLY_LIBS ON CACHE BOOL " " FORCE)
	set(glew-cmake_BUILD_SHARED OFF CACHE BOOL " " FORCE)
	set(glew-cmake_BUILD_STATIC ON CACHE BOOL " " FORCE)
	mark_as_advanced(ONLY_LIBS glew-cmake_BUILD_SHARED glew-cmake_BUILD_STATIC)
	add_subdirectory(dependencies/glew)
	list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glew/include")
	set(GLFRAMEWORK_LIBRARIES ${GLFRAMEWORK_LIBRARIES} opengl32 $<TARGET_OBJECTS:glfw_objects> $<TARGET_OBJECTS:libglew_static> CACHE INTERNAL "")
else()
# Use installed glew
	find_package(GLEW REQUIRED)
	list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
	set(GLFRAMEWORK_LIBRARIES ${GLFRAMEWORK_LIBRARIES} GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects> ${CMAKE_DL_LIBS} CACHE INTERNAL "")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add OpenXR
if (GLFRAMEWORK_BUILD_OPENXR)
	find_package(OpenXR CONFIG QUIET)
	if(OpenXR_FOUND)
		list(APPEND GLFRAMEWORK_INCLUDE_DIRECTORIES OpenXR::Headers)
		list(APPEND GLFRAMEWORK_LIBRARIES OpenXR::openxr_loader)
		list(APPEND GLFRAMEWORK_SOURCES
			"src/_OpenXR.cpp"
			"src/_XrOS.cpp"
		)
	endif()
endif()

# Add CUDA
include(CheckLanguage)
check_language(CUDA)
list(APPEND GLFRAMEWORK_SOURCES
	"src/_OpenGL.cpp"
	"src/_Window.cpp"
	"src/_ImGui.cpp"
	# imgui
	"dependencies/imgui/imgui.cpp"
	"dependencies/imgui/backends/imgui_impl_glfw.cpp"
	"dependencies/imgui/backends/imgui_impl_opengl3.cpp"
	"dependencies/imgui/imgui_draw.cpp"
	"dependencies/imgui/imgui_tables.cpp"
	"dependencies/imgui/imgui_widgets.cpp"
)
if (CMAKE_CUDA_COMPILER AND GLFRAMEWORK_ENABLE_CUDA)
	list(APPEND GLFRAMEWORK_DEFINITIONS -D_CUDA)
	enable_language(CUDA)
	include(FindCUDA/select_compute_arch)
	CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
	string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
	string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
	set(GLFRAMEWORK_CUDA_ARCHITECTURES "" CACHE STRING "Build glframework for a specific GPU architecture.")
	if (DEFINED ENV{GLFRAMEWORK_CUDA_ARCHITECTURES})
		message(STATUS "Obtained target architecture from environment variable GLFRAMEWORK_CUDA_ARCHITECTURES=$ENV{GLFRAMEWORK_CUDA_ARCHITECTURES}")
		set(CMAKE_CUDA_ARCHITECTURES $ENV{GLFRAMEWORK_CUDA_ARCHITECTURES})
	elseif (GLFRAMEWORK_CUDA_ARCHITECTURES)
		message(STATUS "Obtained target architecture from CMake variable GLFRAMEWORK_CUDA_ARCHITECTURES=${GLFRAMEWORK_CUDA_ARCHITECTURES}")
		set(CMAKE_CUDA_ARCHITECTURES ${GLFRAMEWORK_CUDA_ARCHITECTURES})
	else()
		set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
		# set(CMAKE_CUDA_ARCHITECTURES 75)
	endif()
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_EXTENSIONS OFF)
	list(APPEND GLFRAMEWORK_SOURCES "src/_CUDA_NBody_Device.cu")
	list(APPEND GLFRAMEWORK_SOURCES "src/_NBody.cu")
	list(APPEND GLFRAMEWORK_SOURCES "src/_CUDA_Fractal.cu")
	list(APPEND GLFRAMEWORK_SOURCES "src/_Fractal.cu")
endif()
add_library(glframework STATIC ${GLFRAMEWORK_SOURCES})
target_compile_definitions(glframework PUBLIC ${GLFRAMEWORK_DEFINITIONS})
target_include_directories(glframework PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(glframework PUBLIC ${GLFRAMEWORK_LIBRARIES})


if(GLFRAMEWORK_BUILD_DEMOS)
	add_subdirectory(demos/multiSim)
	add_subdirectory(demos/multiView)
	add_subdirectory(demos/multiThread)
	if (CMAKE_CUDA_COMPILER AND GLFRAMEWORK_ENABLE_CUDA)
		add_subdirectory(demos/renderTexture)
	endif()
	if (GLFRAMEWORK_BUILD_OPENXR AND OpenXR_FOUND)
		add_subdirectory(demos/galaxyInHand)
	endif()
endif()