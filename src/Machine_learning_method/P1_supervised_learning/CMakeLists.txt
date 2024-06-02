

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch01_introduction 
	Ch01_Introduction/Introduction.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch01_introduction ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch01_introduction PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch02_perceptron Ch02_Perceptron/Perceptron.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch02_perceptron ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch02_perceptron PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch03_KNN 
	Ch03_KNearestNeighbors/KNearestNeighbors.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch03_KNN ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch03_KNN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch04_naive_bayes 
	Ch04_Naive_Bayes/Naive_Bayes.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch04_naive_bayes ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch04_naive_bayes PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch05_decision_tree 
	Ch05_Decision_tree/Decision_tree.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch05_decision_tree ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch05_decision_tree PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch06_logistic_regression 
	Ch06_Logistic_regression/Logistic_regression.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch06_logistic_regression ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch06_logistic_regression PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch07_SVM 
	Ch07_Support_vector_machine/Support_vector_machine.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch07_SVM ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch07_SVM PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch08_AdaBoost 
	Ch08_AdaBoost/AdaBoost.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch08_AdaBoost ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch08_AdaBoost PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch09_EM 
	Ch09_Expectation_maximization/ExpectationMaximization.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch09_EM ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch09_EM PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch09_GMM 
	Ch09_Expectation_maximization/Gaussian_Mixture_Model.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch09_GMM ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch09_GMM PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P1_supervised_learning_ch10_HMM_baum_welch 
	Ch10_HiddenMarkov/HiddenMarkov_baum_welch.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P1_supervised_learning_ch10_HMM_baum_welch ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P1_supervised_learning_ch10_HMM_baum_welch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------





