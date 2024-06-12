
# -------------------------------------------------------------------------------------

add_executable(ML_P5_appends_Newton_method Newton_method/NewtonMethod.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P5_appends_Newton_method ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P5_appends_Newton_method PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P5_appends_SGD SGD/Stochastic_gradient_descent.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P5_appends_SGD ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P5_appends_SGD PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P5_appends_StochasticGradientDescent SGD/StochasticGradientDescent.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P5_appends_StochasticGradientDescent ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P5_appends_StochasticGradientDescent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

