#include "oidn_denoiser.h"
#include "utils.h"
#include <iostream>
#include <filesystem>
#include <memory>

std::shared_ptr<OidnDenoiser> OidnDenoiser::s_instance = nullptr;

std::shared_ptr<OidnDenoiser> OidnDenoiser::get_instance() {
    if (OidnDenoiser::s_instance.get() == nullptr) {
        OidnDenoiser::s_instance = std::shared_ptr<OidnDenoiser>(new OidnDenoiser());
    }
    return OidnDenoiser::s_instance;
}

void OidnDenoiser::free_instance() {
    if (OidnDenoiser::s_instance.get() != nullptr) {
        OidnDenoiser::s_instance.reset();
    }
}

OidnDenoiser::OidnDenoiser() {
    m_oidn_device = oidn::newDevice(oidn::DeviceType::CUDA);
    m_oidn_device.commit();
    reset_filter();
}

void OidnDenoiser::reset_filter() {
    if (m_filter.getHandle() != nullptr) {
        m_filter.release();
    }
    m_filter = m_oidn_device.newFilter("RT");
    m_mode = OidnMode::NONE;
}

OidnDenoiser::~OidnDenoiser() {
    if (m_filter.getHandle() != nullptr) {
        m_filter.release();
    }
    if (m_oidn_device.getHandle() != nullptr) {
        m_oidn_device.release();
    }
    std::cout << MI_INFO << "OidnDenoiser released successfully" << std::endl;
}

void OidnDenoiser::denoise(float *color, float *output, int width, int height, int channels) {
    unset_and_set_mode(OidnMode::SIMPLE);

    oidn::Format format = oidn::Format(int(oidn::Format::Float) + channels - 1);

    m_filter.setImage("color", color, format, width, height);
    m_filter.setImage("output", output, format, width, height);
    m_filter.set("hdr", true);
    m_filter.commit();
    m_filter.execute();

    check_error(__LINE__);
}

void OidnDenoiser::check_error(const int line) {
    // check errors
    const char *error_message;
    if (m_oidn_device.getError(error_message) != oidn::Error::None) {
        std::cerr << MI_ERROR << "[" << line << "] " << error_message << std::endl;
    }
}

void OidnDenoiser::add_set_weights_task(std::string &weight_path) {
    m_weight_path_to_set = weight_path;
    m_set_weights_task_added = true;
}

void OidnDenoiser::set_weights(std::string &weight_path) {
    if (weight_path.empty()) {
        std::cout << MI_INFO << "Reset weights" << std::endl;
        reset_filter();
        return;
    }
    std::string filename = std::filesystem::path(weight_path).filename().u8string();
    if (weight_path.compare(m_weight_path) == 0) {
        std::cout << MI_INFO << "Weights already loaded" << std::endl;
    } else {
        std::cout << MI_INFO << "Loading weights from " << filename << std::endl
                  << "    " << weight_path << std::endl;
        m_weights = load_file(weight_path);
        m_weight_path = weight_path;
    }
    OidnMode mode = filename.find("alb_nrm") ? OidnMode::AUX : OidnMode::SIMPLE;
    m_mode = mode;
    m_filter.setData("weights", m_weights.data(), m_weights.size());
}

void OidnDenoiser::denoise(float *color, float *normal, float *albedo, float *output, int width, int height, int channels) {
    unset_and_set_mode(OidnMode::AUX);

    oidn::Format format = oidn::Format(int(oidn::Format::Float) + channels - 1);

    m_filter.setImage("color", color, format, width, height);
    m_filter.setImage("normal", normal, format, width, height);
    m_filter.setImage("albedo", albedo, format, width, height);
    m_filter.setImage("output", output, format, width, height);
    m_filter.set("hdr", true);
    m_filter.commit();
    m_filter.execute();

    check_error(__LINE__);
}

void OidnDenoiser::unset_and_set_mode(OidnMode mode) {
    // check weights
    if (m_set_weights_task_added) {
        set_weights(m_weight_path_to_set);
        m_set_weights_task_added = false;
    }

    // check mode
    if (m_mode != mode) {
        m_filter.unsetImage("color");
        m_filter.unsetImage("normal");
        m_filter.unsetImage("albedo");
        m_filter.unsetImage("output");
        m_mode = mode;
    }
}
