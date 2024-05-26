#include <cmath>
#include <vector>
#include <torch/torch.h>
#include <matplot/matplot.h>

using torch::indexing::Slice;
using torch::indexing::None;

using namespace matplot;

int main() {

    std::vector<double> x = {0, 0.0473607, 0.0947214, 0.142082, 0.189443, 0.236803, 0.284164, 0.331525, 0.378886};
    std::cout << "x: " << x << '\n';
    auto y = transform(x, [&](double x) { return cos(x) + rand(0, 1); });
    std::cout << "y: " << y << '\n';
    std::vector<int> c = {1, 1, 1, 2, 2, 2, 3, 3, 3 };
    std::cout << "c: " << c << '\n';

    double sz = 12;
    auto s = scatter(x, y, sz, c); //, std::vector<double>{}, c);
    //s->marker_color({0.f, .5f, .5f});
    //s->marker_face_color({0.f, .7f, .7f});
    s->marker_face(true);
    s->marker_style(line_spec::marker_style::diamond);
    show();

	std::cout << "Done!\n";
	return 0;
}




