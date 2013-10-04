project(pcatool)

cmake_minimum_required(VERSION 2.8)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall -std=c++11")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -std=c++11")

# Includes OpenCV and Boost
find_package( OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect )
find_package( Boost REQUIRED COMPONENTS filesystem system )

# Includes libpca
#file( GLOB LIB_SOURCES /home/ramiro/workspace/libpca-1.2.11/src/*.cpp )
file( GLOB LIB_HEADERS /home/ramiro/workspace/libpca-1.2.11/include/*.h )


# Reference to external lib
add_library( libpca SHARED IMPORTED )
# You can define two import-locations: one for debug and one for release.
set_target_properties( libpca PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/libpca-1.2.11/build/libpca.so.1.2.11 )


add_library( haarcommon-release SHARED IMPORTED )
set_target_properties( haarcommon-release PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build/src/libhaarcommon.so )
add_library( haarcommon-debug SHARED IMPORTED )
set_target_properties( haarcommon-debug PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build-debug/src/libhaarcommon.so )


#include paths to libpca and to haarcommon
include_directories(/home/ramiro/workspace/libpca-1.2.11/include/
                    /home/ramiro/workspace/haarcommon/src/ )
link_directories(/home/ramiro/workspace/haarcommon-build/)


add_executable(pcatool "main.cpp")
target_link_libraries( pcatool libpca ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries( pcatool debug     haarcommon-debug   ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)
target_link_libraries( pcatool optimized haarcommon-release ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

add_executable(haaroptimizer "haaroptimizer.cpp")
target_link_libraries( haaroptimizer libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries( haaroptimizer debug     haarcommon-debug   ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)
target_link_libraries( haaroptimizer optimized haarcommon-release ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)
