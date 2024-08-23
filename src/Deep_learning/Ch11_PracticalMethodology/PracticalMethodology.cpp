/*
 * PracticalMethodology.cpp
 *
 *  Created on: Jul 18, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/csvloader.h"
#include "../../Utils/TempHelpFunctions.h"
#include "../../Algorithms/LogisticRegression.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

namespace F = torch::nn::functional;

torch::Tensor cal_conf_matrix(torch::Tensor labels, torch::Tensor preds) {
    //计算混淆矩阵。
	labels = labels.squeeze();
	preds = preds.squeeze();
    int n_sample = labels.size(0);

    torch::Tensor cm = torch::zeros({2,2}, torch::kInt32);
	int TP = 0, FN = 0, FP = 0, TN = 0;
	for(int i = 0; i < n_sample; i++ ) {
		if(labels[i].data().item<int>() == 1) {
			 if( labels[i].data().item<int>() == preds[i].data().item<int>() )
				 TP += 1;					// y == pred == 1
			 else
				 FN += 1;					// y == 1, pred == 0
		} else {
			if( labels[i].data().item<int>() == preds[i].data().item<int>() )
				TN += 1;					// y == pred == 0
			else
				FP += 1;					// y == 0, pred == 1
		}
	}
	cm[0][0] = TP;
	cm[0][1] = FN;
	cm[1][0] = FP;
	cm[1][1] = TN;
    return cm;
}

std::tuple<double, double, double> cal_PRF1(torch::Tensor labels, torch::Tensor preds) {
    //计算查准率P，查全率R，F1值。

	torch::Tensor cm = cal_conf_matrix(labels, preds);
    double P = cm[0][0].data().item<int>()*1.0/(cm[0][0].data().item<int>()+cm[1][0].data().item<int>());
    double R = cm[0][0].data().item<int>()*1.0/(cm[0][0].data().item<int>()+cm[0][1].data().item<int>());
    double F1 = 2*P*R/(P+R);
    return std::make_tuple(P, R, F1);
}

std::tuple<std::vector<double>, std::vector<double>> cal_PRcurve(torch::Tensor labels, torch::Tensor y_predict) {
	torch::Tensor st_y, st_idx;
	std::tie(st_y, st_idx) = y_predict.sort(0, true);
	torch::Tensor preds = st_y > 0.5;
	preds = preds.to(torch::kInt);
    //计算PR曲线上的值。
	labels = labels.squeeze();
	preds = preds.squeeze();
	st_idx = st_idx.squeeze();

	int n_sample = st_idx.size(0);
	std::vector<double> P, R;
	c10::OptionalArrayRef<long int> dim = {0};
	int num_true = torch::sum(labels.eq(1).to(torch::kInt32), dim).data().item<int>();
	int current_true = 0, current_pred_true = 0;
	for(int i = 0; i < n_sample; i++ ) {
		int idx = st_idx[i].data().item<int>();
		if(labels[idx].data().item<int>() == 1)
			current_true += 1;
		current_pred_true += 1;
		double p = current_true*1.0/current_pred_true;
		double r = current_true*1.0/num_true;
		P.push_back(p);
		R.push_back(r);
	}
    return std::make_tuple(P, R);
}

std::tuple<std::vector<double>, std::vector<double>> cal_ROCcurve(torch::Tensor labels, torch::Tensor y_predict) {

    // 计算ROC曲线上的值。
	torch::Tensor st_y, st_idx;
	std::tie(st_y, st_idx) = y_predict.sort(0, true);
	torch::Tensor preds = st_y > 0.5;
	preds = preds.to(torch::kInt);

	labels = labels.squeeze();
	preds = preds.squeeze();
	st_idx = st_idx.squeeze();

	int n_sample = st_idx.size(0);
	std::vector<double> TPR, FPR;
	int num_true = torch::sum(labels.eq(1).to(torch::kInt32)).data().item<int>();
	int num_false = torch::sum(labels.eq(0).to(torch::kInt32)).data().item<int>();
	int current_true = 0, current_false = 0;
	for(int i = 0; i < n_sample; i++ ) {
		int idx = st_idx[i].data().item<int>();
		if(labels[idx].data().item<int>() == 1)
			current_true += 1;
		if(labels[idx].data().item<int>() == 0)
			current_false += 1;

		double tpr = current_true*1.0/num_true; //# 当前真正例的数量/实际为正的数量
		double fpr = current_false*1.0/num_false;	//# 当前假正例的数量/实际为负的数量
		TPR.push_back(tpr);
		FPR.push_back(fpr);
	}
    return std::make_tuple(TPR, FPR);
}

double area_auc(torch::Tensor labels, torch::Tensor y_predict) {
    // AUC值的梯度法计算
	std::vector<double> TPR, FPR;
    std::tie(TPR, FPR) = cal_ROCcurve(labels, y_predict);
    //# 计算AUC，计算小矩形的面积之和
    double auc = 0.;
    double prev_x = 0;
    int n_sample = TPR.size();
    for( auto& i : range(n_sample, 0)) {
        if(TPR[i] != prev_x ) {
            auc += (TPR[i] - prev_x) * FPR[i];
            prev_x = TPR[i];
        }
    }
    return auc;
}

double naive_auc(torch::Tensor labels, torch::Tensor y_predict) {
    //AUC值的概率法计算
	torch::Tensor st_y, st_idx;
	std::tie(st_y, st_idx) = y_predict.sort(0, false);	//# 对预测概率升序排序
	torch::Tensor preds = st_y > 0.5;
	preds = preds.to(torch::kInt);

	labels = labels.squeeze();
	preds = preds.squeeze();
	st_idx = st_idx.squeeze();

	int n_sample = labels.size(0);
    int n_pos = torch::sum(labels.eq(1).to(torch::kInt32)).data().item<int>();
    int n_neg = n_sample - n_pos;
    int total_pair = n_pos * n_neg;							//# 总的正负样本对的数目
    int count_neg = 0;		//# 统计负样本出现的个数
    int satisfied_pair = 0;	//# 统计满足条件的样本对的个数
    for(int i = 0; i < n_sample; i++ ) {
    	int idx = st_idx[i].data().item<int>();
    	if(labels[idx].data().item<int>() == 1) {
    	    satisfied_pair += count_neg;	//# 表明在这个正样本下，有哪些负样本满足条件
    	} else {
    		count_neg += 1;
    	}
    }

    return(satisfied_pair*1.0 / total_pair);
}

void plot_confusion_matrix(torch::Tensor cm) {
	cm = cm.to(torch::kDouble);

	std::vector<std::vector<double>> C;
	for( int i = 0; i < cm.size(0); i++ ) {
		std::vector<double> c;
		for( int j = 0; j < cm.size(1); j++ ) {
				c.push_back(cm[i][j].item<double>());
		}
		C.push_back(c);
	}
	std::vector<std::string> x_ticks = {"0", "1"};
	std::vector<std::string> y_ticks = {"0", "1"};
	auto h = figure(true);
	h->size(500, 500);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	auto ax = h->nexttile();
	matplot::heatmap(ax, C);
	matplot::colorbar(ax);
    ax->x_axis().ticklabels(x_ticks);
    ax->y_axis().ticklabels(y_ticks);
    matplot::xlabel(ax, "Predicted label");
    matplot::ylabel(ax, "True label");
    matplot::title(ax, "Confusion Matrix");
    matplot::show();
}

void plot_PR_curve(std::vector<double> PP, std::vector<double> RR) {
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	matplot::xlim(ax, {-0.01,1.01});
	matplot::ylim(ax, {-0.01,1.01});
	matplot::plot(ax, RR, PP, "m--")->line_width(3).display_name("P-R");
    matplot::xlabel(ax, "Recall");
    matplot::ylabel(ax, "Precision");
    matplot::title(ax, "Precision-Recall Curve");
    //matplot::legend(ax, {})->location(legend::general_alignment::bottom);
    matplot::show();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Load CSV data\n";
	std::cout << "// --------------------------------------------------\n";
	std::ifstream file;
	std::string path = "./data/breast_cancer_wisconsin.data";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	num_records -= 1; 		// if first row is heads but not record
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// suffle data\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor sidx = RangeTensorIndex(num_records, true);

	std::vector<double> indices = tensorTovector(sidx.squeeze().to(torch::kDouble));
	printVector(indices);
	// ---- split train and test datasets
	std::unordered_set<int> train_idx;

	int train_size = static_cast<int>(num_records * 0.75);
	for( int i = 0; i < train_size; i++ ) {
		train_idx.insert(static_cast<int>(indices.at(i)));
	}

	std::unordered_map<std::string, int> iMap;
	std::cout << "iMap.empty(): " << iMap.empty() << '\n';

	// zscore = true, normalize data = true
	torch::Tensor train_dt, train_lab, test_dt, test_lab;
	std::tie(train_dt, train_lab, test_dt, test_lab) =
			process_split_data2(file, train_idx, iMap, false, false, true, true);

	std::cout << "Train size = " << train_dt.sizes() << "\n";
	std::cout << "Test size = " << test_dt.sizes() << "\n";

	LogisticRegression LR = LogisticRegression(train_dt);
	torch::Tensor w, b;
	std::tie(w, b) = LR.run(train_dt, train_lab);
	torch::Tensor preds =  LR.predict(test_dt);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Confusion Matrix\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor cm = cal_conf_matrix(test_lab, preds);
	std::cout << cm << '\n';
	plot_confusion_matrix(cm);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// precision, recall, and F-score\n";
	std::cout << "// --------------------------------------------------\n";
	double P, R, F1;
	std::tie(P, R, F1) = cal_PRF1(test_lab, preds);
	printf("Precision: %.04f, Recall: %.04f, F-score: %.04f\n", P, R, F1);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// PR curve\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor  y_predict = Sigmoid(torch::mm(test_dt, w) + b);
	std::vector<double> PP, RR;
	auto pr = cal_PRcurve(test_lab, y_predict.clone());
	PP = std::get<0>(pr);
	RR = std::get<1>(pr);
	printVector(PP);
	printVector(RR);
	plot_PR_curve(PP, RR);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// ROC curve and AUC value\n";
	std::cout << "// --------------------------------------------------\n";
	double area_AUC = area_auc(test_lab, y_predict.clone());
	double naive_AUC = naive_auc(test_lab, y_predict.clone());
	printf("Area AUC: %.04f\tNaive AUC: %.04f\n", area_AUC, naive_AUC);

	std::vector<double> TPR, FPR;
	std::tie(TPR, FPR) = cal_ROCcurve(test_lab, y_predict.clone());
	printVector(PP);
	printVector(RR);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax = F->nexttile();
	matplot::hold(ax, true);
	auto s = matplot::scatter(ax, FPR, TPR, 8.0);
	s->marker_color("b");
	s->marker_face_color({0, .5, .5});
	s->display_name("TPR-FPR");
	matplot::plot(ax, FPR, TPR, "c-.")->line_width(2);
	matplot::plot(ax, {0.,1.0}, {0., 1.0}, "r--")->line_width(3);
    matplot::xlabel(ax, "False Positive Rate");
    matplot::ylabel(ax, "True Positive Rate");
    matplot::xlim(ax, {-0.01,1.01});
    matplot::ylim(ax, {-0.01,1.01});
    matplot::title(ax, "Receiver Operating Characteristic");
    matplot::hold(ax, false);
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}
