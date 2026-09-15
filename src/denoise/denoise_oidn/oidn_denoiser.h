#pragma once

#include <OpenImageDenoise/oidn.hpp>
#include <vector>
#include <string>
#include <memory>

enum OidnMode {
    NONE,
    SIMPLE,
    AUX
};

class OidnDenoiser {
private:
    OidnDenoiser();
    void unset_and_set_mode(OidnMode mode);
    void set_weights(std::string &weight_path);
    void reset_filter();
    void check_error(const int line);

public:
    ~OidnDenoiser();
    void denoise(float *color, float *output, int width, int height, int channels);
    void denoise(float *color, float *normal, float *albedo, float *output, int width, int height, int channels);
    void add_set_weights_task(std::string &weight_path);

    static std::shared_ptr<OidnDenoiser> get_instance();
    static void free_instance();

private:
    static std::shared_ptr<OidnDenoiser> s_instance;
    oidn::DeviceRef m_oidn_device;
    oidn::FilterRef m_filter;
    OidnMode m_mode = OidnMode::NONE;
    std::string m_weight_path{};
    std::vector<char> m_weights{};

    // delayed task, set weights when denoise() is called
    bool m_set_weights_task_added = false;
    std::string m_weight_path_to_set{};
};