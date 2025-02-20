
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_1_Backpropagation_in_toy_model
	C7_1_Backpropagation_in_toy_model.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_1_Backpropagation_in_toy_model ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_1_Backpropagation_in_toy_model PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_2_Backpropagation
	C7_2_Backpropagation.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_2_Backpropagation ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_2_Backpropagation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_3_Initialization
	C7_3_Initialization.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_3_Initialization ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_3_Initialization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_4_Simple_example
	C7_4_Simple_example.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_4_Simple_example ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_4_Simple_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------









