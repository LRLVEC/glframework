cmake_minimum_required(VERSION 3.18)

project(glframework
	LANGUAGES CXX
)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TINY_STL_BUILD_TESTS OFF CACHE BOOL " " FORCE)
mark_as_advanced(TINY_STL_BUILD_TESTS)

option(GLFRAMEWORK_BUILD_TESTS "Build glframework test programs" ON)


if(GLFRAMEWORK_BUILD_TESTS)
	add_subdirectory(tests)
endif()