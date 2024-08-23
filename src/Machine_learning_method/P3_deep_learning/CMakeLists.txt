

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch23_FFN Ch23_Feedforward_networks/feedforward_nn.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch23_FFN ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch23_FFN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch23_NeuralNetMLP Ch23_Feedforward_networks/NeuralNetMLP.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/fashion.cpp
	../../Utils/fashion.h)
				
target_link_libraries(ML_P3_deep_learning_Ch23_NeuralNetMLP ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch23_NeuralNetMLP PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch24_CNN_classifying Ch24_CNN/CNN_classifying.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp
	../../Utils/fashion.cpp
	../../Utils/fashion.h)
				
target_link_libraries(ML_P3_deep_learning_Ch24_CNN_classifying ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch24_CNN_classifying PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch24_CNN_text_classification Ch24_CNN/CNN_text_classification.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	)
		
target_link_libraries(ML_P3_deep_learning_Ch24_CNN_text_classification ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch24_CNN_text_classification PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch25_Sequences Ch25_RNN/Sequences.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch25_Sequences ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch25_Sequences PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch25_LSTMTagger Ch25_RNN/LSTMTagger.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch25_LSTMTagger ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch25_LSTMTagger PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch26_CNN_seq2seq Ch26_Seq2seq/CNN_seq2seq.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch26_CNN_seq2seq ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch26_CNN_seq2seq PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch26_LSTM_seq2seq Ch26_Seq2seq/LSTM_seq2seq.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch26_LSTM_seq2seq ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch26_LSTM_seq2seq PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch27_AutoEncoder Ch27_Pretrain_NLP_model/AutoEncoder.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp
	../../Utils/fashion.cpp
	../../Utils/fashion.h)
				
target_link_libraries(ML_P3_deep_learning_Ch27_AutoEncoder ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch27_AutoEncoder PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch28_Zero_sum_game Ch28_GAN/Zero_sum_game.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch28_Zero_sum_game ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch28_Zero_sum_game PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch28_FashionImgGenerateUsingGAN Ch28_GAN/FashionImgGenerateUsingGAN.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp
	../../Utils/fashion.cpp
	../../Utils/fashion.h)
				
target_link_libraries(ML_P3_deep_learning_Ch28_FashionImgGenerateUsingGAN ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch28_FashionImgGenerateUsingGAN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------








