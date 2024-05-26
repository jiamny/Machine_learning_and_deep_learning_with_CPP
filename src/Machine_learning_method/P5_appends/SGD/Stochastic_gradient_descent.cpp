/*
 * A_stochastic_gradient_descent.cpp
 *
 *  Created on: May 3, 2024
 *      Author: jiamny
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>

double f(double x) {
	return x * x;
}

double golden_section_for_line_search(double (*func)(double), double a0, double b0, double epsilon) {
    /*一维搜索极小值点（黄金分割法）

    :param func: [function] 一元函数
    :param a0: [int/float] 目标区域左侧边界
    :param b0: [int/float] 目标区域右侧边界
    :param epsilon: [int/float] 精度
    */
    double a1 = a0 + 0.382 * (b0 - a0);
    double b1 = b0 - 0.382 * (b0 - a0);
    double fa = func(a1);
    double fb = func(b1);

    while( (b1 - a1) > epsilon ) {
        if( fa <= fb ) {
            b0 = b1;
            b1 = a1;
            fb = fa;
            a1 = a0 + 0.382 * (b0 - a0);
            fa = func(a1);
        } else {
            a0 = a1;
            a1 = b1;
            fa = fb;
            b1 = b0 - 0.382 * (b0 - a0);
            fb = func(b1);
        }
    }
    return (a1 + b1) / 2;
}

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	std::cout << "golden_section_for_line_search: " <<
			golden_section_for_line_search(f, -10, 5, 1e-6) << '\n';  // 5.263005013597177e-06

	return 0;
}


