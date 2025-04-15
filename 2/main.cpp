#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::filesystem;

namespace {
double parse_exposure(const string &str) {
    auto slash = str.find('/');

    if (slash != string::npos) {
        double numerator = stod(str.substr(0, slash));
        double denominator = stod(str.substr(slash + 1));
        return numerator / denominator;
    }

    return stod(str);
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: assignment-2 <directory>\n";
        return EXIT_FAILURE;
    }

    fs::path dir(argv[1]);

    ifstream csv(dir / "exposure_times.csv");
    string line;
    getline(csv, line);

    vector<double> exposure_times;
    vector<cv::Mat> images;

    while (getline(csv, line)) {
        istringstream ss(line);

        string name, exposure;

        if (getline(ss, name, ';') && getline(ss, exposure)) {
            double exposure_ = parse_exposure(exposure);
            exposure_times.push_back(exposure_);

            string path((dir / name).string());
            cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

            if (image.empty()) {
                cerr << "Failed to load image " << path << '\n';
                return EXIT_FAILURE;
            }

            images.push_back(image);
        }
    }

    vector<array<double, 3>> curve;
    curve.reserve(256);

    ifstream curve_f(dir / "curve.m");
    getline(curve_f, line);

    while (getline(curve_f, line)) {
        if (line.starts_with(']')) {
            break;
        }

        istringstream ss(line);
        string r_s, g_s, b_s;

        if (getline(ss, r_s, ' ') && getline(ss, g_s, ' ') && getline(ss, b_s)) {
            curve.push_back({stod(b_s), stod(g_s), stod(r_s)});
        }
    }

    cv::Size size(images[0].size());
    cv::Mat hdr_image(size, CV_32FC3);

    // Mounting HDR image from loaded images
    for (auto i = 0; i < size.height; ++i) {
        for (auto j = 0; j < size.width; ++j) {
            array<double, 3> irradiances = {0, 0, 0};
            array<size_t, 3> irradiance_count = {0, 0, 0};

            for (size_t im = 0; im < images.size(); ++im) {
                const auto &image = images[im];
                auto pixel = image.at<cv::Vec3b>(i, j);

                for (size_t p = 0; p < 3; ++p) {
                    auto z = pixel[p];
                    auto curve_value = curve[z][p];
                    auto actual_exposure = std::exp(curve_value);
                    auto irradiance = actual_exposure / exposure_times[im];

                    if (z > 0 && z < UINT8_MAX) {
                        irradiances[p] += irradiance;
                        irradiance_count[p] += 1;
                    }
                }
            }

            auto &pixel = hdr_image.at<cv::Vec3f>(i, j);
            for (size_t p = 0; p < 3; ++p) {
                float irradiance =
                    irradiance_count[p] > 0 ? irradiances[p] / irradiance_count[p] : 0.0f;
                pixel[p] = irradiance;
            }
        }
    }

    cv::imwrite("hdr_image.hdr", hdr_image);

    // Luminance image
    cv::Mat luminance_image(size, CV_32FC1);
    double total_luminance = 0.0;
    size_t count = 0;

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            auto hdr_pixel = hdr_image.at<cv::Vec3f>(i, j);
            auto luminance = hdr_pixel[2] * 0.299f + hdr_pixel[1] * 0.587f + hdr_pixel[0] * 0.114f;

            luminance_image.at<float>(i, j) = luminance;

            if (luminance > 0.0) {
                total_luminance += std::log(luminance + std::numeric_limits<double>::epsilon());
                count += 1;
            }
        }
    }

    double average_luminance = std::exp(total_luminance / static_cast<double>(count));

    cv::imwrite("hdr_luminance.hdr", luminance_image);

    // Global tone mapping
    constexpr double ALPHA = 0.18;

    cv::Mat global_tone_mapped(size, hdr_image.type());

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            float luminance = luminance_image.at<float>(i, j);
            float scaled_luminance = ALPHA * luminance / average_luminance;
            float global_operator = scaled_luminance / (1 + scaled_luminance);

            float scale = luminance > 0.0f ? (global_operator / luminance) : 0.0f;
            cv::Vec3f pixel = hdr_image.at<cv::Vec3f>(i, j) * scale;
            global_tone_mapped.at<cv::Vec3f>(i, j) = pixel;
        }
    }

    cv::imwrite("ldr_image.jpg", global_tone_mapped);

    return EXIT_SUCCESS;
}
