Usage:
1. add ```target_include_directories(your_target PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})```.
2. add ```target_link_libraries(your_target PUBLIC ${GLFRAMEWORK_LIBRARIES})```.
3. if you don't wish to build tests in your project, add ```set(GLFRAMEWORK_BUILD_TESTS OFF CACHE BOOL " " FORCE)```