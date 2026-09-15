#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <cassert>

#ifndef DIMENSIONS_OUT
#define DIMENSIONS_OUT DIMENSIONS
#endif


void launch_query_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    const torch::Tensor depths_tensor,
    const torch::Tensor resolutions_tensor,
    const torch::Tensor entries_tensor,
    int64_t hashmap_size,
    int disp_stride,
    const std::vector<torch::Tensor>& features,
    torch::Tensor result
);

std::vector<torch::Tensor> launch_query_backward(
    const torch::Tensor grad_output,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    const torch::Tensor depths_tensor,
    const torch::Tensor resolutions_tensor,
    const torch::Tensor entries_tensor,
    int64_t hashmap_size,
    int disp_stride,
    const std::vector<torch::Tensor>& features
);

void launch_accumulate_forward(
    const torch::Tensor feature,
    const torch::Tensor level_weights,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    std::vector<torch::Tensor>& features,
    std::vector<torch::Tensor>& counts
);


torch::Tensor query_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    const torch::Tensor depths_tensor,
    const torch::Tensor resolutions_tensor,
    const torch::Tensor entries_tensor,
    int64_t hashmap_size,
    const std::vector<torch::Tensor>& features
) {
    auto N = dir.size(0);
    auto L = features.size();
    auto D = features[0].size(-1);
    auto options = features[0].options();

    assert(L == LEVELS);
    assert(D == DIMENSIONS_OUT);
    assert(pos.size(0) == N);
    assert(pos.size(1) == 3);
    assert(disp.size(0) == N);
    const bool disp_is_shared = disp.dim() == 1 || (disp.dim() == 2 && disp.size(1) == 1);
    const bool disp_is_per_level = disp.dim() == 2 && disp.size(1) == LEVELS;
    assert(disp_is_shared || disp_is_per_level);
    assert(children.size() >= LEVELS);
    assert(grid_scales_tensor.size(0) == LEVELS);
    assert(grid_scales_tensor.size(1) == 3);
    assert(depths_tensor.size(0) == LEVELS);
    assert(depths_tensor.scalar_type() == torch::kInt32);
    assert(resolutions_tensor.size(0) == LEVELS);
    assert(resolutions_tensor.scalar_type() == torch::kInt32);
    assert(entries_tensor.size(0) == LEVELS);
    assert(entries_tensor.scalar_type() == torch::kInt64);
    assert(hashmap_size > 0);

    const int disp_stride = disp_is_shared ? 0 : static_cast<int>(disp.size(1));

    torch::Tensor result = torch::zeros({N, L * D}, options);

    launch_query_forward(
        pos,
        children,
        dir,
        disp,
        grid_scales_tensor,
        depths_tensor,
        resolutions_tensor,
        entries_tensor,
        hashmap_size,
        disp_stride,
        features,
        result
    );

    return result;
}

std::vector<torch::Tensor> query_backward(
    const torch::Tensor grad_output,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    const torch::Tensor depths_tensor,
    const torch::Tensor resolutions_tensor,
    const torch::Tensor entries_tensor,
    int64_t hashmap_size,
    const std::vector<torch::Tensor>& features
) {
    auto N = dir.size(0);
    auto L = features.size();
    auto D = features[0].size(-1);

    assert(L == LEVELS);
    assert(D == DIMENSIONS_OUT);
    assert(grad_output.size(0) == N);
    assert(grad_output.size(1) == LEVELS * DIMENSIONS_OUT);
    assert(pos.size(0) == N);
    assert(pos.size(1) == 3);
    assert(disp.size(0) == N);
    const bool disp_is_shared = disp.dim() == 1 || (disp.dim() == 2 && disp.size(1) == 1);
    const bool disp_is_per_level = disp.dim() == 2 && disp.size(1) == LEVELS;
    assert(disp_is_shared || disp_is_per_level);
    assert(children.size() >= LEVELS);
    assert(grid_scales_tensor.size(0) == LEVELS);
    assert(grid_scales_tensor.size(1) == 3);
    assert(depths_tensor.size(0) == LEVELS);
    assert(depths_tensor.scalar_type() == torch::kInt32);
    assert(resolutions_tensor.size(0) == LEVELS);
    assert(resolutions_tensor.scalar_type() == torch::kInt32);
    assert(entries_tensor.size(0) == LEVELS);
    assert(entries_tensor.scalar_type() == torch::kInt64);
    assert(hashmap_size > 0);
    assert(grad_output.scalar_type() == dir.scalar_type());
    assert(pos.scalar_type() == dir.scalar_type());
    assert(disp.scalar_type() == dir.scalar_type());

    const int disp_stride = disp_is_shared ? 0 : static_cast<int>(disp.size(1));

    return launch_query_backward(
        grad_output.contiguous(),
        pos,
        children,
        dir,
        disp,
        grid_scales_tensor,
        depths_tensor,
        resolutions_tensor,
        entries_tensor,
        hashmap_size,
        disp_stride,
        features
    );
}

void accumulate_forward(
    const torch::Tensor feature,
    const torch::Tensor level_weights,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    std::vector<torch::Tensor>& features,
    std::vector<torch::Tensor>& counts
) {
    auto N = feature.size(0);
    auto L = features.size();
    auto D = features[0].size(-1);

    assert(L == LEVELS);
    assert(D == DIMENSIONS);
    assert(pos.size(0) == N);
    assert(pos.size(1) == 3);
    assert(feature.size(1) == LEVELS);
    assert(feature.size(2) == DIMENSIONS);
    assert(level_weights.size(0) == N);
    assert(level_weights.size(1) == LEVELS);
    assert(level_weights.scalar_type() == feature.scalar_type());
    assert(pos.scalar_type() == feature.scalar_type());
    assert(dir.scalar_type() == feature.scalar_type());
    assert(disp.scalar_type() == feature.scalar_type());
    assert(children.size() == LEVELS);
    assert(grid_scales_tensor.size(0) == LEVELS);
    assert(grid_scales_tensor.size(1) == 3);

    launch_accumulate_forward(feature, level_weights, pos, children, dir, disp, grid_scales_tensor, features, counts);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query_forward", &query_forward,
        py::arg("pos"),
        py::arg("children"),
        py::arg("dir"),
        py::arg("disp"),
        py::arg("grid_scales"),
        py::arg("depths"),
        py::arg("resolutions"),
        py::arg("entries"),
        py::arg("hashmap_size"),
        py::arg("features"),
        "OctaOctree query forward (CUDA)"
    );
    m.def("query_backward", &query_backward,
        py::arg("grad_output"),
        py::arg("pos"),
        py::arg("children"),
        py::arg("dir"),
        py::arg("disp"),
        py::arg("grid_scales"),
        py::arg("depths"),
        py::arg("resolutions"),
        py::arg("entries"),
        py::arg("hashmap_size"),
        py::arg("features"),
        "OctaOctree query backward (CUDA)"
    );
    m.def("accumulate_forward", &accumulate_forward,
        py::arg("feature"),
        py::arg("level_weights"),
        py::arg("pos"),
        py::arg("children"),
        py::arg("dir"),
        py::arg("disp"),
        py::arg("grid_scales"),
        py::arg("features"),
        py::arg("counts"),
        "OctaOctree accumulate forward (CUDA)"
    );
}
