# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")

if (CMAKE_CUDA_COMPILER AND GLFRAMEWORK_ENABLE_CUDA)
# if(FALSE)
	enable_language(CUDA)
	add_executable(test tests.cu ../dependencies/imgui/imgui_demo.cpp)
else()
	add_executable(test tests.cpp ../dependencies/imgui/imgui_demo.cpp)
endif()
target_include_directories(test PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(test PUBLIC ${GLFRAMEWORK_LIBRARIES})
target_link_libraries(test PUBLIC glframework)
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
message("Copy /shaders")
add_custom_command(TARGET test PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test>/shaders
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:test>/shaders/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:test>/shaders
	COMMAND_EXPAND_LISTS
)

