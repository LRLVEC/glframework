# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -O2 -Wall")
add_executable(galaxyInHand galaxyInHand.cpp)
set_target_properties(galaxyInHand PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/galaxyInHand")
target_include_directories(galaxyInHand PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(galaxyInHand PUBLIC ${GLFRAMEWORK_LIBRARIES})
target_link_libraries(galaxyInHand PUBLIC glframework)

file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
message("Copy shaders")
add_custom_command(TARGET galaxyInHand PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:galaxyInHand>/shaders
	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:galaxyInHand>/shaders/*
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADERS} $<TARGET_FILE_DIR:galaxyInHand>/shaders
	COMMAND_EXPAND_LISTS
)

# file(GLOB RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/resources")
# message("Copy resources")
# add_custom_command(TARGET galaxyInHand PRE_BUILD
# 	COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:galaxyInHand>/resources
# 	# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS} $<TARGET_FILE_DIR:galaxyInHand>/resources/*
# 	COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES} $<TARGET_FILE_DIR:galaxyInHand>/resources
# 	COMMAND_EXPAND_LISTS
# )

