#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "oidn_denoiser.h"
#include "utils.h"
#include <pybind11/pybind11.h>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void init() {
    std::shared_ptr<OidnDenoiser> oidn_denoiser = OidnDenoiser::get_instance();
    std::cout << MI_INFO << "OidnDenoiser initialized successfully" << std::endl;
}

void denoise(torch::Tensor color, torch::Tensor output, int width, int height, int channels) {
    if (channels > 3 || channels < 1) {
        std::cerr << MI_ERROR << " Oidn only supports up to 3 channels" << std::endl;
        return;
    }

    std::shared_ptr<OidnDenoiser> oidn_denoiser = OidnDenoiser::get_instance();
    oidn_denoiser->denoise((float *)color.data_ptr(), (float *)output.data_ptr(), width, height, channels);
}

void denoise_with_normal_and_albedo(torch::Tensor color, torch::Tensor normal, torch::Tensor albedo, torch::Tensor output, int width, int height, int channels) {
    if (channels > 3 || channels < 1) {
        std::cerr << "[Error] Oidn only supports up to 3 channels" << std::endl;
        return;
    }

    std::shared_ptr<OidnDenoiser> oidn_denoiser = OidnDenoiser::get_instance();
    oidn_denoiser->denoise((float *)color.data_ptr(), (float *)normal.data_ptr(), (float *)albedo.data_ptr(), (float *)output.data_ptr(), width, height, channels);
}

void set_weights(std::string &weight_path) {
    std::shared_ptr<OidnDenoiser> oidn_denoiser = OidnDenoiser::get_instance();
    oidn_denoiser->add_set_weights_task(weight_path);
}

void free_denoiser() {
    std::shared_ptr<OidnDenoiser> oidn_denoiser = OidnDenoiser::get_instance();
    oidn_denoiser->free_instance();
}

PYBIND11_MODULE(setup_oidn_example, m) {
    m.doc() = "pybind11 oidn plugin";  // optional module docstring
    m.def("init", &init, "Initialize oidn denoiser");
    m.def("denoise", &denoise, "Denoise image");
    m.def("denoise_with_normal_and_albedo", &denoise_with_normal_and_albedo, "Denoise image with normal and albedo");
    m.def("set_weights", &set_weights, "Set weights");
    m.def("free_denoiser", &free_denoiser, "Free oidn denoiser");
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}