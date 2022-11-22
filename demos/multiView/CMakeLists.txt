# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")
add_executable(multiView multiView.cpp)
set_target_properties(multiView PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/multiView")
target_include_directories(multiView PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(multiView PUBLIC ${GLFRAMEWORK_LIBRARIES})
target_link_libraries(multiView PUBLIC glframework)
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
message("Copy /shaders")
add_custom_command(TARGET multiView PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:multiView>/shaders
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:multiView>/shaders/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:multiView>/shaders
	COMMAND_EXPAND_LISTS
)

