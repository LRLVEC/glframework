# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	add_executable(test tests.cu)
	set(CMAKE_CUDA_ARCHITECTURES 86)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_EXTENSIONS OFF)
	message("Found CUDA! Use cuda version.")
else()
	add_executable(test tests.cpp)
	message("CUDA not found! Use non-cuda version.")
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

