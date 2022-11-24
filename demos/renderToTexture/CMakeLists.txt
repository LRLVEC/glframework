# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")
add_executable(renderToTexture renderToTexture.cpp)
set_target_properties(renderToTexture PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/renderToTexture")
target_include_directories(renderToTexture PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(renderToTexture PUBLIC ${GLFRAMEWORK_LIBRARIES})
target_link_libraries(renderToTexture PUBLIC glframework)
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
message("Copy shaders")
add_custom_command(TARGET renderToTexture PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:renderToTexture>/shaders
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:renderToTexture>/shaders/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:renderToTexture>/shaders
	COMMAND_EXPAND_LISTS
)
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/resources")
message("Copy resources")
add_custom_command(TARGET renderToTexture PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:renderToTexture>/resources
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:renderToTexture>/resources/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:renderToTexture>/resources
	COMMAND_EXPAND_LISTS
)

