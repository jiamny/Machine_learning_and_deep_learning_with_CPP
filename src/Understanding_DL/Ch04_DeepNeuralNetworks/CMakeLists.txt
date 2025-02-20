
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch04_1_ComposingNeuralNetworks
	C4_1_ComposingNeuralNetworks.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch04_1_ComposingNeuralNetworks ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch04_1_ComposingNeuralNetworks  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch04_2_Clipping_functions
	C4_2_Clipping_functions.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch04_2_Clipping_functions ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch04_2_Clipping_functions PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch04_3_Deep_Networks
	C4_3_Deep_Networks.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch04_3_Deep_Networks ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch04_3_Deep_Networks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------






