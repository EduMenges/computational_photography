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

    if (!csv.is_open()) {
        cerr << "Could not open `exposure_times.csv` file\n";
        return EXIT_FAILURE;
    }

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
            cv::Mat image = cv::imread(path, cv::IMREAD_ANYCOLOR);

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

    if (!curve_f.is_open()) {
        cerr << "Could not open `curve.m` file\n";
        return EXIT_FAILURE;
    }

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
    cv::Mat hdr_image(size, CV_64FC3);

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

                    if (z > 0 && z < UINT8_MAX) {
                        auto curve_value = curve[z][p];
                        auto actual_exposure = std::exp(curve_value);
                        auto irradiance = actual_exposure / exposure_times[im];

                        irradiances[p] += irradiance;
                        irradiance_count[p] += 1;
                    }
                }
            }

            auto &pixel = hdr_image.at<cv::Vec3d>(i, j);
            for (size_t p = 0; p < 3; ++p) {
                double irradiance = irradiance_count[p] > 0
                                        ? irradiances[p] / static_cast<double>(irradiance_count[p])
                                        : 0.0f;
                pixel[p] = irradiance;
            }
        }
    }

    cv::Mat hdr_f32_image;
    hdr_image.convertTo(hdr_f32_image, CV_32FC3);
    cv::imwrite("hdr_image.hdr", hdr_f32_image);

    // Luminance image
    cv::Mat luminance_image(size, CV_64FC1);
    double total_luminance = 0.0;
    size_t count = 0;

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            auto hdr_pixel = hdr_image.at<cv::Vec3d>(i, j);
            auto luminance = hdr_pixel[2] * 0.299 + hdr_pixel[1] * 0.587 + hdr_pixel[0] * 0.114;

            luminance_image.at<double>(i, j) = luminance;

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
            double luminance = luminance_image.at<double>(i, j);
            double scaled_luminance = (ALPHA / average_luminance) * luminance;
            double global_operator = scaled_luminance / (1 + scaled_luminance);

            double scale = luminance > 0.0 ? (global_operator / luminance) : 0.0;
            cv::Vec3d pixel = hdr_image.at<cv::Vec3d>(i, j) * scale;
            global_tone_mapped.at<cv::Vec3d>(i, j) = pixel;
        }
    }

    // Convert to 16 bit RGB
    cv::Mat ldr_image;
    global_tone_mapped.convertTo(ldr_image, CV_16UC3, 65535.0);
    cv::imwrite("ldr_image.png", ldr_image);

    // Gamma correction
    cv::Mat gamma_corrected;
    cv::pow(global_tone_mapped, 1.0 / 2.2, global_tone_mapped);
    global_tone_mapped.convertTo(gamma_corrected, CV_16UC3, 65535.0);
    cv::imwrite("ldr_image_gamma_corrected.png", gamma_corrected);

    return EXIT_SUCCESS;
}
