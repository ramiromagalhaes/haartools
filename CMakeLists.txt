project(pcatool)

cmake_minimum_required(VERSION 2.8)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

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


#include paths to libpca and to haarcommon
include_directories(/home/ramiro/workspace/libpca-1.2.11/include/
                    /home/ramiro/workspace/haarcommon/ )
link_directories(/home/ramiro/workspace/haarcommon-build/)


add_executable(pcatool "main.cpp")
target_link_libraries( pcatool libpca haarcommon ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(haaroptimizer "haaroptimizer.cpp")
target_link_libraries( haaroptimizer haarcommon libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES})
