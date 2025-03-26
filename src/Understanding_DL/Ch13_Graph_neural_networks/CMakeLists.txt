
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch13_1_Graph_representation
	C13_1_Graph_representation.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch13_1_Graph_representation ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch13_1_Graph_representation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch13_2_Graph_classification
	C13_2_Graph_classification.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch13_2_Graph_classification ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch13_2_Graph_classification PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch13_4_Graph_attention_networks
	C13_4_Graph_attention_networks.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch13_4_Graph_attention_networks ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch13_4_Graph_attention_networks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------










