# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if (CUDA_FOUND)
	add_executable(test tests.cu)
else()
	add_executable(test tests.cpp)
endif()
target_include_directories(test PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(test PUBLIC ${GLFRAMEWORK_LIBRARIES})
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
message("Copy /shaders")
add_custom_command(TARGET test PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test>/shaders
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:test>/shaders/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:test>/shaders
	COMMAND_EXPAND_LISTS
)

