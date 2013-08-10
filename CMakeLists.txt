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


# Another external lib. Not sure if this is a nice way to do that...
file( GLOB LIB_SOURCES /home/ramiro/workspace/ecrsgen/lib/*.cpp )
file( GLOB LIB_HEADERS /home/ramiro/workspace/ecrsgen/lib/*.h )
add_library( haarwavelet ${LIB_SOURCES} ${LIB_HEADERS} )


aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries( ${PROJECT_NAME} libpca haarwavelet ${OpenCV_LIBS} ${Boost_LIBRARIES})
