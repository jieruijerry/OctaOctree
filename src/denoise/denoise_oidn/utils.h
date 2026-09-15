#pragma once
#include <vector>
#include <string>

#define RED_BEGIN "\033[31m"
#define RED_END "\033[0m"
#define GREEN_BEGIN "\033[32m"
#define GREEN_END "\033[0m"
#define YELLOW_BEGIN "\033[33m"
#define YELLOW_END "\033[0m"

#define MI_ERROR "\033[31m[ERROR]\033[0m "
#define MI_INFO "\033[32m[INFO]\033[0m "
#define MI_WARNING "\033[33m[WARNING]\033[0m "

std::vector<char> load_file(const std::string &filename);
