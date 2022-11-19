Usage:
1. add these to your CMakeLists.txt:
```
get_target_property(GLFRAMEWORK_INCLUDE_DIRECTORIES glframework INCLUDE_DIRECTORIES)
target_include_directories(your_target PUBLIC ${GLFRAMEWORK_INCLUDE_DIRECTORIES})
target_link_libraries(your_target PUBLIC ${GLFRAMEWORK_LIBRARIES})
```
2. if you don't wish to build tests in your project, add ```set(GLFRAMEWORK_BUILD_TESTS OFF CACHE BOOL " " FORCE)```