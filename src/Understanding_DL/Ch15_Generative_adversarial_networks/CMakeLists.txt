
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch15_1_GAN_toy_example
	C15_1_GAN_toy_example.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch15_1_GAN_toy_example ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch15_1_GAN_toy_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch15_2_Wasserstein_distance
	C15_2_Wasserstein_distance.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch15_2_Wasserstein_distance ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch15_2_Wasserstein_distance PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------







