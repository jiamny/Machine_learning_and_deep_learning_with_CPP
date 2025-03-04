
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch11_1_Shattered_gradients
	C11_1_Shattered_gradients.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch11_1_Shattered_gradients ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch11_1_Shattered_gradients PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch11_2_Residual_Networks
	C11_2_Residual_Networks.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch11_2_Residual_Networks ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch11_2_Residual_Networks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch11_3_Batch_normalization
	C11_3_Batch_normalization.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch11_3_Batch_normalization ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch11_3_Batch_normalization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------











