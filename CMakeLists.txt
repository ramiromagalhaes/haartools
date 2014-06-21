project(haartools)

cmake_minimum_required(VERSION 2.8)

# Debug and release configurations
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_DEBUG -Wextra -Wall -std=c++11")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -std=c++11")

# Includes OpenCV and Boost
find_package( OpenCV REQUIRED COMPONENTS core highgui imgproc )
find_package( Boost REQUIRED COMPONENTS filesystem system )

# Includes libpca
include_directories( /home/ramiro/workspace/libpca-1.2.11/include/ )
add_library( libpca SHARED IMPORTED )
set_target_properties( libpca PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/libpca-1.2.11/build/libpca.so.1.2.11 )

# Includes haarcommon
include_directories( /home/ramiro/workspace/haarcommon/src/ )
add_library( haarcommon-release SHARED IMPORTED )
set_target_properties( haarcommon-release PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build/src/libhaarcommon.so )
add_library( haarcommon-debug SHARED IMPORTED )
set_target_properties( haarcommon-debug   PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/haarcommon-build-debug/src/libhaarcommon.so )

# Include training database tools library
include_directories( /home/ramiro/workspace/trainingdatabase-tools/lib )
add_library( trainingdatabase SHARED IMPORTED )
set_target_properties( trainingdatabase PROPERTIES IMPORTED_LOCATION /home/ramiro/workspace/trainingdatabase-tools-build/lib/libtrainingdatabase-lib.so )


# The Haar wavelet generator
add_executable( haargen haargen.cpp )
target_link_libraries( haargen haarcommon-release ${OpenCV_LIBS} ${Boost_LIBRARIES})

# The Haar wavelet checker
add_executable( haarcheck haarcheck.cpp )
target_link_libraries( haarcheck haarcommon-release ${OpenCV_LIBS} )

# The Haar wavelet PCA optimizer
add_executable(haaroptimizer haaroptimizer.cpp mypca.h mypca.cpp optimization_commons.h )
target_link_libraries( haaroptimizer debug     haarcommon-debug   trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( haaroptimizer optimized haarcommon-release trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

# The Haar wavelet PCA optimizer for the second experiment
add_executable(haaroptimizer2 haaroptimizer2.cpp mypca.h mypca.cpp optimization_commons.h )
target_link_libraries( haaroptimizer2 debug     haarcommon-debug   trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( haaroptimizer2 optimized haarcommon-release trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

# The Haar wavelet PCA optimizer for an alternative to the second experiment
add_executable(haaroptimizer2a haaroptimizer2a.cpp mypca.h mypca.cpp optimization_commons.h )
target_link_libraries( haaroptimizer2a debug     haarcommon-debug   trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( haaroptimizer2a optimized haarcommon-release trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

# Checks if the Haar wavelets optimized for the second experiment are okay
add_executable( haarcheck2 haarcheck2.cpp )
target_link_libraries( haarcheck2 haarcommon-release )

# The Haar wavelet PCA optimizer for the third experiment
add_executable(haaroptimizer3 haaroptimizer3.cpp mypca.h mypca.cpp optimization_commons.h )
target_link_libraries( haaroptimizer3 debug     haarcommon-debug   trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( haaroptimizer3 optimized haarcommon-release trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )

# The Haar wavelets for the Adhikari's default experiment
add_executable(haaroptimizer_adhikari haaroptimizer_adhikari.cpp mypca.h mypca.cpp optimization_commons.h )
target_link_libraries( haaroptimizer_adhikari debug     haarcommon-debug   trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( haaroptimizer_adhikari optimized haarcommon-release trainingdatabase libpca armadillo tbb ${OpenCV_LIBS} ${Boost_LIBRARIES} )
