
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch10_1_1D_Convolution
	C10_1_1D_Convolution.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch10_1_1D_Convolution ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch10_1_1D_Convolution PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch10_2_Convolution_for_MNIST1D
	C10_2_Convolution_for_MNIST1D.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch10_2_Convolution_for_MNIST1D ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch10_2_Convolution_for_MNIST1D PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch10_3_2D_Convolution
	C10_3_2D_Convolution.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch10_3_2D_Convolution ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch10_3_2D_Convolution PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch10_4_Down_and_Up_sampling
	C10_4_Down_and_Up_sampling.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch10_4_Down_and_Up_sampling ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch10_4_Down_and_Up_sampling PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch10_5_Convolution_for_MNIST
	C10_5_Convolution_for_MNIST.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp
)

# link libraries
target_link_libraries( UDL_Ch10_5_Convolution_for_MNIST ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch10_5_Convolution_for_MNIST PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------








