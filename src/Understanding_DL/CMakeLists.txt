
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch01_BackgroundMaths 
	Ch01_BackgroundMaths/BackgroundMaths.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch01_BackgroundMaths ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch01_BackgroundMaths  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch02_SupervisedLearning 
	Ch02_SupervisedLearning/SupervisedLearning.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch02_SupervisedLearning ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch02_SupervisedLearning  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworks_I 
	Ch03_ShallowNeuralNetworks/ShallowNetworks_I.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworks_I ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworks_I  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworks_II 
	Ch03_ShallowNeuralNetworks/ShallowNetworks_II.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworks_II ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworks_II  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ActivationFunctions 
	Ch03_ShallowNeuralNetworks/ActivationFunctions.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ActivationFunctions ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ActivationFunctions  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworkRegions 
	Ch03_ShallowNeuralNetworks/ShallowNetworkRegions.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworkRegions ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworkRegions  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch04_1_ComposingNeuralNetworks
	Ch04_DeepNeuralNetworks/C4_1_ComposingNeuralNetworks.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch04_1_ComposingNeuralNetworks ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch04_1_ComposingNeuralNetworks  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch04_2_Clipping_functions
	Ch04_DeepNeuralNetworks/C4_2_Clipping_functions.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch04_2_Clipping_functions ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch04_2_Clipping_functions PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch04_3_Deep_Networks
	Ch04_DeepNeuralNetworks/C4_3_Deep_Networks.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch04_3_Deep_Networks ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch04_3_Deep_Networks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch05_1_Least_Squares_Loss
	Ch05_Loss_functions/C5_1_Least_Squares_Loss.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch05_1_Least_Squares_Loss ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch05_1_Least_Squares_Loss PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch05_2_Binary_Cross_Entropy_Loss
	Ch05_Loss_functions/C5_2_Binary_Cross_Entropy_Loss.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch05_2_Binary_Cross_Entropy_Loss ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch05_2_Binary_Cross_Entropy_Loss PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch05_3_Multiclass_Cross_entropy_Loss
	Ch05_Loss_functions/C5_3_Multiclass_Cross_entropy_Loss.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch05_3_Multiclass_Cross_entropy_Loss ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch05_3_Multiclass_Cross_entropy_Loss PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_1_Line_Search
	Ch06_Fitting_models/C6_1_Line_Search.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_1_Line_Search ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_1_Line_Search PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_2_Gradient_Descent
	Ch06_Fitting_models/C6_2_Gradient_Descent.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_2_Gradient_Descent ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_2_Gradient_Descent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_3_Stochastic_Gradient_Descent
	Ch06_Fitting_models/C6_3_Stochastic_Gradient_Descent.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_3_Stochastic_Gradient_Descent ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_3_Stochastic_Gradient_Descent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_4_Momentum
	Ch06_Fitting_models/C6_4_Momentum.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_4_Momentum ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_4_Momentum PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch06_5_Adam
	Ch06_Fitting_models/C6_5_Adam.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch06_5_Adam ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch06_5_Adam PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_1_Backpropagation_in_toy_model
	Ch07_Gradients_and_initialization/C7_1_Backpropagation_in_toy_model.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_1_Backpropagation_in_toy_model ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_1_Backpropagation_in_toy_model PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_2_Backpropagation
	Ch07_Gradients_and_initialization/C7_2_Backpropagation.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_2_Backpropagation ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_2_Backpropagation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_3_Initialization
	Ch07_Gradients_and_initialization/C7_3_Initialization.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_3_Initialization ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_3_Initialization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch07_4_Simple_example
	Ch07_Gradients_and_initialization/C7_4_Simple_example.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch07_4_Simple_example ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch07_4_Simple_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------






