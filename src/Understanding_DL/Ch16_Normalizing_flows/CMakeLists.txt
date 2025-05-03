
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch16_1_1D_normalizing_flows
	C16_1_1D_normalizing_flows.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch16_1_1D_normalizing_flows ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch16_1_1D_normalizing_flows PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------







