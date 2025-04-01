#include <iostream>
#include <libraw/libraw.h>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./assignment-1 <input.dng>\n";
        return EXIT_FAILURE;
    }

    const char *file_name = argv[1];

    // Loading image
    LibRaw processor;
    if (processor.open_file(file_name) != LIBRAW_SUCCESS) {
        std::cerr << "Error opening RAW file!" << std::endl;
        return EXIT_FAILURE;
    }

    if (processor.unpack() != LIBRAW_SUCCESS) {
        std::cerr << "Error unpacking RAW file!" << std::endl;
        return EXIT_FAILURE;
    }

    processor.raw2image();

    int height = processor.imgdata.sizes.iheight;
    int width = processor.imgdata.sizes.iwidth;
    auto black = processor.imgdata.color.black;

    cv::Mat bayer_image(height, width, CV_16UC1);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            uint16_t color =
                _sat_sub_u16(processor.imgdata.image[i * width + j][processor.COLOR(i, j)], black);
            bayer_image.at<uint16_t>(i, j) = color;
        }
    }

    // Demosaicking
    cv::Mat demosaicked_image;
    cv::cvtColor(bayer_image, demosaicked_image, cv::COLOR_BayerBG2BGR);

    // Normalization
    cv::Mat normalized_image;
    cv::normalize(demosaicked_image, normalized_image, 0, UINT16_MAX, cv::NORM_MINMAX);

    // White balance
    std::vector<cv::Mat> channels;
    cv::split(normalized_image, channels);

    cv::Scalar avg = cv::mean(normalized_image);
    double meanVal = (avg[0] + avg[1] + avg[2]) / 3.0;

    for (int i = 0; i < 3; i++) {
        channels[i] = channels[i] * (meanVal / avg[i]);
    }

    cv::Mat balanced_image;
    cv::merge(channels, balanced_image);

    // Gamma correction
    constexpr double GAMMA = 2.0;
    cv::Mat img_norm, img_gamma;
    balanced_image.convertTo(img_norm, CV_32F, 1.0 / 65535.0);
    cv::pow(img_norm, 1.0 / GAMMA, img_gamma);
    img_gamma.convertTo(img_gamma, CV_16UC3, 65535.0);

    cv::imwrite("bayer_image.png", bayer_image);
    cv::imwrite("demosaicked_image.png", demosaicked_image);
    cv::imwrite("normalized_image.png", normalized_image);
    cv::imwrite("balanced_image.png", balanced_image);
    cv::imwrite("gamma_corrected.png", img_gamma);

    return EXIT_SUCCESS;
}
