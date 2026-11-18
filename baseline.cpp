#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "baseline.hpp"


std::uint64_t checksum(const cv::Mat& image) {
    if (image.type() != CV_8UC1) {
        throw std::invalid_argument("checksum expects a CV_8UC1 image");
    }

    std::uint64_t hash = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;

    for (int y = 0; y < image.rows; ++y) {
        const std::uint8_t* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            hash ^= row[x];
            hash *= prime;
        }
    }

    return hash;
}

int main(int argc, char** argv) {
    // Check number of arguments
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " [thread_count] [video_path] [iterations] [ - | display ]\n";
        return 1;
    }

    int threadCount = 0;
    try {
        threadCount = std::stoi(argv[1]);
    } catch (const std::exception&) {
        std::cout << "Invalid number of threads: " << argv[1] << "\n";
        return 1;
    }
    if (threadCount <= 0) {
        std::cout << "Invalid number of threads: " << threadCount << "\n";
        return 1;
    }

    const std::string defaultPath = "../../bbb_sunflower_2160p_60fps_normal.mp4";
    const std::string videoPath = argc > 2 ? argv[2] : defaultPath;

    const int iterations = argc > 3 ? std::stoi(argv[3]) : 1;

    // Output mode: no arg = window, "-" = headless with no files, otherwise display window
    const bool display = (argc > 4) && std::string(argv[4]) == "display";

    cv::VideoCapture video(videoPath);
    if (!video.isOpened()) {
        std::cout << "Could not open video: " << videoPath << "\n";
        return 1;
    }

    // Set single frame and read it
    video.set(cv::CAP_PROP_POS_FRAMES, 21120);
    cv::Mat frame;
    video.read(frame);
    const int rows = frame.rows;
    const int cols = frame.cols;

    cv::Mat sobelOutput(rows - 2, cols - 2, CV_8UC1);

    if (display) {
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
    }

    using namespace operations;
    operations::RowWorkerPool workers(threadCount);
    cv::Mat refSobel, diff;
    operations::ErrorStats errorStats;
    uint64_t refPixels = 0;

    for (int i = 0; i < iterations; ++i) {
        if (frame.empty()) break;

        workers.runRows(rows - 2, [&](int firstRow, int lastRow) {
            operations::processSegment(frame, sobelOutput, firstRow, lastRow);
        });
        
        if (display) {
            cv::imshow("Output", sobelOutput);
            cv::waitKey(1);
        }
    }

    // Single-threaded reference and error (outside all timed sections)
    cv::Mat gray_fixed(rows, cols, CV_8UC1);
    refSobel.create(rows - 2, cols - 2, CV_8UC1);
    
    // --- Fixed-point grayscale (Q8) using Mat::at<> ---
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const cv::Vec3b& p = frame.at<cv::Vec3b>(y, x);
            int v = (operations::kRedWeight * p[2] + operations::kGreenWeight * p[1] + operations::kBlueWeight * p[0]
                 + operations::kGrayRounding) >> operations::kGrayShift; // Q8 -> 8-bit
            gray_fixed.at<uchar>(y, x) = static_cast<uchar>(v);
        }
    }
    
    // --- Approximate Sobel magnitude: |gx| + |gy| using Mat::at<> ---
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int top_left  = gray_fixed.at<uchar>(y - 1, x - 1);
            int top_mid   = gray_fixed.at<uchar>(y - 1, x);
            int top_right = gray_fixed.at<uchar>(y - 1, x + 1);

            int mid_left  = gray_fixed.at<uchar>(y, x - 1);
            int mid_right = gray_fixed.at<uchar>(y, x + 1);

            int bot_left  = gray_fixed.at<uchar>(y + 1, x - 1);
            int bot_mid   = gray_fixed.at<uchar>(y + 1, x);
            int bot_right = gray_fixed.at<uchar>(y + 1, x + 1);

            int gx = (top_right - top_left) +
                            2 * (mid_right - mid_left) +
                            (bot_right - bot_left);
            int gy = (bot_left + 2 * bot_mid + bot_right) -
                            (top_left + 2 * top_mid + top_right);

            int mag = std::abs(gx) + std::abs(gy); // approximate magnitude
            if (mag > 255) mag = 255;
            refSobel.at<uchar>(y - 1, x - 1) = static_cast<uchar>(mag);
        }
    }
    
    refPixels = refSobel.total();

    operations::accumulateError(sobelOutput, refSobel, diff, errorStats);

    // Output statistics
    if (iterations > 1) {
        const double meanError = errorStats.pixels
            ? errorStats.sumAbsDiff / errorStats.pixels : 0.0;
        const bool checksumMatch = checksum(sobelOutput) == checksum(refSobel);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Configuration: optimized_" << threadCount << "T\n";
        std::cout << "Thread count: " << threadCount << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Loop Checksum: " << checksum(sobelOutput) << "\n";
        std::cout << "Reference Checksum: " << checksum(refSobel) << "\n";
        std::cout << "Checksums Match: " << (checksumMatch ? "Yes" : "No") << "\n";

        std::cout << std::setprecision(6);
        std::cout << "Mean absolute error: " << meanError << "\n";
        std::cout << "Max difference: " << errorStats.maxDiff << "\n";
        std::cout << "Differing pixels: " << errorStats.differingPixels
                  << " / " << refPixels << "\n";
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}
