# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")
add_executable(multiThread multiThread.cpp ../../dependencies/imgui/imgui_demo.cpp)
set_target_properties(multiThread PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/multiThread")
target_include_directories(multiThread PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(multiThread PUBLIC ${GLFRAMEWORK_LIBRARIES})
target_link_libraries(multiThread PUBLIC glframework)
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
message("Copy shaders")
add_custom_command(TARGET multiThread PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:multiThread>/shaders
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:multiThread>/shaders/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:multiThread>/shaders
	COMMAND_EXPAND_LISTS
)

