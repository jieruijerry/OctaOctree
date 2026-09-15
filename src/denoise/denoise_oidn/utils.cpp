#include "utils.h"
#include <fstream>
#include <iostream>

std::vector<char> load_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.fail()) {
        std::cerr << RED_BEGIN << "cannot open file: '" << filename << "'" << RED_END << std::endl;
    }
    file.seekg(0, file.end);
    const size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    if (file.fail()) {
        std::cerr << RED_BEGIN << "error reading from file: '" << filename << "'" << RED_END << std::endl;
    }
    return buffer;
}