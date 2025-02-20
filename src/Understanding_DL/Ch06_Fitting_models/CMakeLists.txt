
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_1_Line_Search
	C6_1_Line_Search.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_1_Line_Search ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_1_Line_Search PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_2_Gradient_Descent
	C6_2_Gradient_Descent.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_2_Gradient_Descent ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_2_Gradient_Descent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_3_Stochastic_Gradient_Descent
	C6_3_Stochastic_Gradient_Descent.cpp
	../../Utils/helpfunction.h
	../../Utils/UDL_util.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_3_Stochastic_Gradient_Descent ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_3_Stochastic_Gradient_Descent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_4_Momentum
	C6_4_Momentum.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_4_Momentum ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_4_Momentum PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_5_Adam
	C6_5_Adam.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_5_Adam ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_5_Adam PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------







