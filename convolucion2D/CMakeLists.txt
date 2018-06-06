cmake_minimum_required(VERSION 3.5)
project( DisplayImage )
add_definitions(-std=c++11)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_52 -std=c++11 )
cuda_add_executable( convoConstMem convoConstMem.cu )
target_link_libraries( convoConstMem ${OpenCV_LIBS} )

