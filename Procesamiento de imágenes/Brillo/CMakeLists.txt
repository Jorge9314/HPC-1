cmake_minimum_required(VERSION 2.8)
project( Brillo )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
include_directories( ${OpenCV_INCLUDE_DIRS} )
cuda_add_executable( a.out brightness.cu )
target_link_libraries( a.out ${OpenCV_LIBS} )


