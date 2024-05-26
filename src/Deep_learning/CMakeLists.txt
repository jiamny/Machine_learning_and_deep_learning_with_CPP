
# -------------------------------------------------------------------------------------
# add executable
add_executable( DL_Ch02_check_matplotplusplus Ch02_Linear_algebra/Check_matplotplusplus.cpp)

# link libraries
target_link_libraries( DL_Ch02_check_matplotplusplus ${TORCH_LIBRARIES} matplot)
set_target_properties( DL_Ch02_check_matplotplusplus  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch02_Linear_algebra Ch02_Linear_algebra/Linear_algebra.cpp)
				
target_link_libraries(DL_Ch02_Linear_algebra ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch02_Linear_algebra PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch03_Probability
		Ch03_Probability_and_information_theory/Probability.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch03_Probability ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch03_Probability PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch03_Information_theory 
		Ch03_Probability_and_information_theory/Information_theory.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch03_Information_theory ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch03_Information_theory PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch04_Numerical_computation 
		Ch04_Numerical_computation/Numerical_computation.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch04_Numerical_computation ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch04_Numerical_computation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch05_Machine_learning_basics 
		Ch05_Machine_learning_basics/Machine_learning_basics.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch05_Machine_learning_basics ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch05_Machine_learning_basics PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch05_Supervised_learning_algorithms 
		Ch05_Machine_learning_basics/Supervised_learning_algorithms.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch05_Supervised_learning_algorithms ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch05_Supervised_learning_algorithms PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch05_Unsupervised_learning_algorithms 
		Ch05_Machine_learning_basics/Unsupervised_learning_algorithms.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch05_Unsupervised_learning_algorithms ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch05_Unsupervised_learning_algorithms PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch06_MLP 
		Ch06_Deep_feedforward_networks/MLP.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch06_MLP ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch06_MLP PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch06_XOR 
		Ch06_Deep_feedforward_networks/XOR.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch06_XOR ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch06_XOR PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch07_Regularization
		Ch07_Regularization/Regularization.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch07_Regularization ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_Regularization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch07_dropout
		Ch07_Regularization/dropout.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch07_dropout ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_dropout PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch07_weight_decay
		Ch07_Regularization/weight_decay.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch07_weight_decay ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_weight_decay PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(DL_Ch07_dataset_augmentation 
		Ch07_Regularization/Dataset_augmentation.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch07_dataset_augmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_dataset_augmentation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------




