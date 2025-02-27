cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Machine_learning_and_deep_learning_with_CPP VERSION 1.0.0 LANGUAGES CXX)

# ------------------------------------------------------------------------------------------------------
# 	Embed Matplot++ as subdirectory
#
# https://alandefreitas.github.io/matplotplusplus/integration/cmake/embed-as-subdirectory/
# ------------------------------------------------------------------------------------------------------
add_subdirectory(matplotplusplus)


set(PYTORCH_VERSION "2.6.0")

set(USE_CPU_ONLY NO)


if( NOT USE_CPU_ONLY )

	find_package(PythonInterp REQUIRED)
	
	if(NOT DEFINED CUDA_ARCHITECTURES)
		# change based on your GPU. check at: https://developer.nvidia.com/cuda-gpus
   		set(CMAKE_CUDA_ARCHITECTURES 86 87 CACHE STRING "CUDA architectures" FORCE)
	endif()
	
	# change libtorch path
	set( LIBTORCH_PATH "/home/hhj/Open_libs/libtorch_cu118" )
	
	
	set( CUDA_HOME /usr/local/cuda-11.8 )
	set( CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8 )
	set( CUDA_CUDART_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so )
	set( CUDA_nvrtc_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so )
	set( CUDA_curand_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcurand.so )
	set( CUDA_cufft_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcufft.so )
	set( CUDA_cublas_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so )
	set( CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc )
	# set( CUDA_cublas_device_LIBRARY /usr/local/cuda-11.8/lib64 )
	set( CUDA_INCLUDE_DIRS /usr/local/cuda-11.8/include )
	# SET(CMAKE_CUDA_COMPILER_ENV_VAR /usr/local/cuda-11.8/bin/nvcc)

	set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch" )
	#--------------------------------------------------------------------------------------------
	# following line solve the issue of Can't link a project using cmake with OpenCV and LibTorch
	#-------------------------------------------------------------------------------------------- 
	list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
             
	set(CMAKE_MODULE_PATH "${LIBTORCH_PATH}/share/cmake/Torch") 
	 
	# add torch package
	find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}") 
else()
	# change libtorch path
	set( LIBTORCH_PATH "/home/hhj/libtorch_cpu" )
	
	set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch" )
	#--------------------------------------------------------------------------------------------
	# following line solve the issue of Can't link a project using cmake with OpenCV and LibTorch
	#-------------------------------------------------------------------------------------------- 
	list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
             
	set(CMAKE_MODULE_PATH "${LIBTORCH_PATH}/share/cmake/Torch") 
	   
	# add torch package
	find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS "${LIBTORCH_PATH}") 
endif()

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Torch library status:")
message(STATUS "    version: ${Torch_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}") 


# add opencv package to the project
FIND_PACKAGE( OpenCV 4.8.0 REQUIRED PATHS "/usr/local/")
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ) 
SET(OpenCV_LIBRARIES ${OpenCV_LIBS})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_FOUND:${OpenCV_FOUND}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lm -ldl")

if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(fetch_libtorch)
endif()

INCLUDE_DIRECTORIES( ${TORCH_INCLUDE_DIRS} )

# Add  sub-projects:
add_subdirectory("src/Deep_learning")	
add_subdirectory("src/Machine_learning_method")
add_subdirectory("src/Understanding_DL")



