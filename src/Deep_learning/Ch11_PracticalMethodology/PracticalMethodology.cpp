/*
 * PracticalMethodology.cpp
 *
 *  Created on: Jul 18, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>

#include "../../Utils/helpfunction.h"
#include "../../Utils/TempHelpFunctions.h"

using torch::indexing::Slice;
using torch::indexing::None;

#include <matplot/matplot.h>
using namespace matplot;

namespace F = torch::nn::functional;

/*
 def cal_conf_matrix(labels, preds) {
    //计算混淆矩阵。

    n_sample = len(labels)
    result = pd.DataFrame(index=range(0,n_sample),columns=('probability','label'))
    result['label'] = np.array(labels)
    result['probability'] = np.array(preds)
    cm = np.arange(4).reshape(2,2)
    cm[0,0] = len(result[result['label']==1][result['probability']>=0.5]) # TP，注意这里是以 0.5 为阈值
    cm[0,1] = len(result[result['label']==1][result['probability']<0.5])  # FN
    cm[1,0] = len(result[result['label']==0][result['probability']>=0.5]) # FP
    cm[1,1] = len(result[result['label']==0][result['probability']<0.5])  # TN
    return cm
}

 */

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	std::cout << "Done!\n";
	return 0;
}
