
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworks_I 
	ShallowNetworks_I.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworks_I ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworks_I  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworks_II 
	ShallowNetworks_II.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworks_II ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworks_II  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ActivationFunctions 
	ActivationFunctions.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ActivationFunctions ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ActivationFunctions  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworkRegions 
	ShallowNetworkRegions.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworkRegions ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworkRegions  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------










