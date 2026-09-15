#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <cassert>


#define CONCAT  0
#define MEAN    1
#define INTERP  2

#ifndef LAYER_REDUCE
#define LAYER_REDUCE CONCAT
#endif


void launch_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
);

void launch_forward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
);

torch::Tensor forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids
) {
    auto N = pos.size(0);
    auto L = grids.size();
    auto D = grids[0].size(2);
    auto options = grids[0].options();

    assert(L == LEVELS);
    assert(D == DIMENSIONS);

    torch::Tensor result;
    if (LAYER_REDUCE == MEAN) {
        result = torch::zeros({N, D}, options);
    } else if (LAYER_REDUCE == CONCAT) {
        result = torch::zeros({N, L * D}, options);
    } else {
        throw std::invalid_argument("Invalid layer reduce option" + LAYER_REDUCE);
    }

    launch_forward(pos, grids, result);

    return result;
}

torch::Tensor forward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const std::vector<torch::Tensor>& grids
) {
    auto N = pos.size(0);
    auto L = grids.size();
    auto D = grids[0].size(2);
    auto options = grids[0].options();

    assert(L == LEVELS);
    assert(D == DIMENSIONS);

    torch::Tensor result = torch::zeros({N, D}, options);

    launch_forward_layer_interp(pos, point_size, grids, result);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
        py::arg("position"),
        py::arg("grids"),
        "Multi-resolution hash grid with layer reduction (CUDA)"
    );
    m.def("forward_layer_interp", &forward_layer_interp,
        py::arg("position"),
        py::arg("point_size"),
        py::arg("grids"),
        "Multi-resolution hash grid with layer interpolation (CUDA)"
    );
}