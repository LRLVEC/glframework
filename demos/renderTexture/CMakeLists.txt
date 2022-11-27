# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")
add_executable(renderTexture renderTexture.cpp)
set_target_properties(renderTexture PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/renderTexture")
target_include_directories(renderTexture PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(renderTexture PUBLIC ${GLFRAMEWORK_LIBRARIES})
target_link_libraries(renderTexture PUBLIC glframework)
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
message("Copy shaders")
add_custom_command(TARGET renderTexture PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:renderTexture>/shaders
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:renderTexture>/shaders/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:renderTexture>/shaders
	COMMAND_EXPAND_LISTS
)
file(GLOB RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/resources")
message("Copy resources")
add_custom_command(TARGET renderTexture PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:renderTexture>/resources
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:renderTexture>/resources/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES} $<TARGET_FILE_DIR:renderTexture>/resources
	COMMAND_EXPAND_LISTS
)

