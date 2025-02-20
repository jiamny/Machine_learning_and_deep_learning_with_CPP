
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch08_1_MNIST_1D_Performance
	C8_1_MNIST_1D_Performance.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch08_1_MNIST_1D_Performance ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch08_1_MNIST_1D_Performance PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch08_2_Bias_variance_trade_off
	C8_2_Bias_variance_trade_off.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch08_2_Bias_variance_trade_off ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch08_2_Bias_variance_trade_off PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch08_3_Double_descent
	C8_3_Double_descent.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch08_3_Double_descent ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch08_3_Double_descent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch08_4_High_dimensional_spaces
	C8_4_High_dimensional_spaces.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch08_4_High_dimensional_spaces ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch08_4_High_dimensional_spaces PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------









