set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")
add_executable(test tests.cpp)
target_include_directories(test PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(test PUBLIC ${GLFRAMEWORK_LIBRARIES})
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")
add_custom_command(TARGET test POST_BUILD
	COMMAND mkdir -p $<TARGET_FILE_DIR:test>/shaders
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:test>/shaders
	COMMAND_EXPAND_LISTS
)