#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "NIS_Config.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

class NISConfigWrapper {
private:
    NISConfig m_nis_config;

public:
    bool pyNVScalerUpdateConfig(float sharpness,
                                uint32_t inputTextureWidth, uint32_t inputTextureHeight,
                                uint32_t outputViewportWidth, uint32_t outputViewportHeight,
                                NISHDRMode hdrMode = NISHDRMode::None) {
        return NVScalerUpdateConfig(m_nis_config, sharpness,
                                    0, 0, inputTextureWidth, inputTextureHeight,
                                    inputTextureWidth, inputTextureHeight,
                                    0, 0, outputViewportWidth, outputViewportHeight,
                                    outputViewportWidth, outputViewportHeight,
                                    hdrMode);
    }

    bool pyNVSharpenUpdateConfig(float sharpness,
                                 uint32_t inputTextureWidth, uint32_t inputTextureHeight,
                                 NISHDRMode hdrMode = NISHDRMode::None) {
        return NVSharpenUpdateConfig(m_nis_config, sharpness,
                                     0, 0, inputTextureWidth, inputTextureHeight,
                                     inputTextureWidth, inputTextureHeight,
                                     0, 0,
                                     hdrMode);
    }

    int get_size() {
        return sizeof(m_nis_config);
    }

    pybind11::array_t<float> get_config() {
        const int size = sizeof(m_nis_config);
        pybind11::array_t<float> result(size);
        pybind11::buffer_info buf = result.request();
        memcpy(buf.ptr, &m_nis_config, size);
        return result;
    }

    // TODO: static will cause errors
    pybind11::array_t<uint16_t> get_coef_scale_fp16() {
        const int size = sizeof(coef_scale_fp16);
        const int eles = size / sizeof(uint16_t);
        pybind11::array_t<uint16_t> result(eles);
        pybind11::buffer_info buf = result.request();
        memcpy(buf.ptr, &coef_scale_fp16, size);
        return result;
    }

    pybind11::array_t<uint16_t> get_coef_usm_fp16() {
        const int size = sizeof(coef_usm_fp16);
        const int eles = size / sizeof(uint16_t);
        pybind11::array_t<uint16_t> result(eles);
        pybind11::buffer_info buf = result.request();
        memcpy(buf.ptr, &coef_usm_fp16, size);
        return result;
    }

    pybind11::array_t<uint16_t> get_shader_params() {
        std::vector<uint16_t> data = {(uint16_t)kFilterSize, (uint16_t)kPhaseCount};
        const int eles = data.size();
        const int size = sizeof(uint16_t) * eles;
        pybind11::array_t<uint16_t> result(eles);
        pybind11::buffer_info buf = result.request();
        memcpy(buf.ptr, data.data(), size);
        return result;
    }

    pybind11::array_t<uint32_t> get_optimal_dispatch_size() {
        NISOptimizer opt(true, NISGPUArchitecture::NVIDIA_Generic);

        std::vector<uint32_t> data = {opt.GetOptimalBlockWidth(), opt.GetOptimalBlockHeight(), opt.GetOptimalThreadGroupSize()};
        const int eles = data.size();
        const int size = sizeof(uint32_t) * eles;
        pybind11::array_t<uint32_t> result(eles);
        pybind11::buffer_info buf = result.request();
        memcpy(buf.ptr, data.data(), size);
        return result;
    }
};

PYBIND11_MODULE(pyNISConfigWrapper, m) {
    pybind11::enum_<NISHDRMode>(m, "NISHDRMode")
        .value("None", NISHDRMode::None)
        .value("Linear", NISHDRMode::Linear)
        .value("PQ", NISHDRMode::PQ)
        .export_values();

    pybind11::class_<NISConfigWrapper>(m, "pyNISConfig")
        .def(pybind11::init<>())
        .def("NVScalerUpdateConfig", &NISConfigWrapper::pyNVScalerUpdateConfig, pybind11::arg("sharpness"),
             pybind11::arg("inputTextureWidth"), pybind11::arg("inputTextureHeight"),
             pybind11::arg("outputViewportWidth"), pybind11::arg("outputViewportHeight"),
             pybind11::arg("hdrMode") = NISHDRMode::None)
        .def("NVSharpenUpdateConfig", &NISConfigWrapper::pyNVSharpenUpdateConfig, pybind11::arg("sharpness"),
             pybind11::arg("inputTextureWidth"), pybind11::arg("inputTextureHeight"),
             pybind11::arg("hdrMode") = NISHDRMode::None)
        .def("get_size", &NISConfigWrapper::get_size)
        .def("get_config", &NISConfigWrapper::get_config)
        .def("get_coef_scale_fp16", &NISConfigWrapper::get_coef_scale_fp16)
        .def("get_coef_usm_fp16", &NISConfigWrapper::get_coef_usm_fp16)
        .def("get_optimal_dispatch_size", &NISConfigWrapper::get_optimal_dispatch_size)
        .def("get_shader_params", &NISConfigWrapper::get_shader_params);
}