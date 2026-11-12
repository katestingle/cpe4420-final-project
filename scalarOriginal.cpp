#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>

// Error statistics for Sobel output comparison
struct ErrorStats {
    double sumAbsDiff = 0.0;
    uint64_t pixels = 0;
    uint64_t differingPixels = 0;
    int maxDiff = 0;
};

// Accumulate error statistics between the test and reference Sobel outputs
void accumulateError(const cv::Mat& test, const cv::Mat& ref,
                     cv::Mat& diff, ErrorStats& stats) {
    cv::absdiff(test, ref, diff);
    double frameMax = 0.0;
    cv::minMaxLoc(diff, nullptr, &frameMax);
    stats.maxDiff = std::max(stats.maxDiff, static_cast<int>(frameMax));
    stats.sumAbsDiff += cv::sum(diff)[0];
    stats.differingPixels += static_cast<uint64_t>(cv::countNonZero(diff));
    stats.pixels += static_cast<uint64_t>(diff.total());
}

int main(int argc, char** argv) {
    // Check number of arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " [video_path] [iterations] [ - | display ]\n";
        return 1;
    }
    
    const std::string defaultPath = "../../bbb_sunflower_2160p_60fps_normal.mp4";
    const std::string videoPath = (argc > 1) ? argv[1] : defaultPath;
    const int iterations = std::stoi(argv[2]);
    const bool display = (argc > 3) && std::string(argv[3]) == "display";

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
    
    // Mats for optimized (fixed-point)
    cv::Mat gray_fixed(rows, cols, CV_8UC1);
    cv::Mat sobel_approx(rows - 2, cols - 2, CV_8UC1);

    // Display window for optimized result only
    if (display) {
        cv::namedWindow("Output", cv::WINDOW_NORMAL);
    }

    // Error metrics
    uint64_t totalPixels = 0;
    ErrorStats errorStats;
    cv::Mat referenceGray, referenceSobel, diff;
    
    // Fixed-point coefficients (Q8)
    // Approximating BT.709: 0.2126, 0.7152, 0.0722
    // scale = 256 (shift = 8). Coefficients chosen: round(coeff*256)
    const int shift = 8;
    const int wR = 54;   // round(0.2126 * 256) = 54
    const int wG = 183;  // round(0.7152 * 256) = 183
    const int wB = 18;   // round(0.0722 * 256) = 18
    const int rounding = 1 << (shift - 1); // for rounding before shift

    // Put all variables outside of loop
    int r, g, b, v;
    int top_left, top_mid, top_right;
    int mid_left, mid_right;
    int bot_left, bot_mid, bot_right;
    int gx, gy, mag;

    // Process frames
    for (int i = 0; i < iterations; ++i) {
        if (frame.empty()) break;

        // --- Fixed-point grayscale (Q8) using Mat::at<> ---
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const cv::Vec3b& p = frame.at<cv::Vec3b>(y, x);
                r = p[2];
                g = p[1];
                b = p[0];
                v = (wR * r + wG * g + wB * b + rounding) >> shift; // Q8 -> 8-bit
                gray_fixed.at<uchar>(y, x) = static_cast<uchar>(v);
            }
        }
        
        // --- Approximate Sobel magnitude: |gx| + |gy| using Mat::at<> ---
        for (int y = 1; y < rows - 1; ++y) {
            for (int x = 1; x < cols - 1; ++x) {
                top_left  = gray_fixed.at<uchar>(y - 1, x - 1);
                top_mid   = gray_fixed.at<uchar>(y - 1, x);
                top_right = gray_fixed.at<uchar>(y - 1, x + 1);

                mid_left  = gray_fixed.at<uchar>(y, x - 1);
                mid_right = gray_fixed.at<uchar>(y, x + 1);

                bot_left  = gray_fixed.at<uchar>(y + 1, x - 1);
                bot_mid   = gray_fixed.at<uchar>(y + 1, x);
                bot_right = gray_fixed.at<uchar>(y + 1, x + 1);

                gx = (top_right - top_left) +
                               2 * (mid_right - mid_left) +
                               (bot_right - bot_left);
                gy = (bot_left + 2 * bot_mid + bot_right) -
                               (top_left + 2 * top_mid + top_right);

                mag = std::abs(gx) + std::abs(gy); // approximate magnitude
                if (mag > 255) mag = 255;
                sobel_approx.at<uchar>(y - 1, x - 1) = static_cast<uchar>(mag);
            }
        }
        
        totalPixels += static_cast<uint64_t>(sobel_approx.total());

        if (display) {
            // Display only the optimized result
            cv::imshow("Output", sobel_approx);
            cv::waitKey(1);
        }
    }

    // Reference and error (outside all timed sections)
    referenceGray.create(rows, cols, CV_8UC1);
    referenceSobel.create(rows - 2, cols - 2, CV_8UC1);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const cv::Vec3b& p = frame.at<cv::Vec3b>(y, x);
            r = p[2];
            g = p[1];
            b = p[0];
            v = (wR * r + wG * g + wB * b + rounding) >> shift; // Q8 -> 8-bit
            referenceGray.at<uchar>(y, x) = static_cast<uchar>(v);
        }
    }

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            top_left = referenceGray.at<uchar>(y - 1, x - 1);
            top_mid = referenceGray.at<uchar>(y - 1, x);
            top_right = referenceGray.at<uchar>(y - 1, x + 1);
            mid_left = referenceGray.at<uchar>(y, x - 1);
            mid_right = referenceGray.at<uchar>(y, x + 1);
            bot_left = referenceGray.at<uchar>(y + 1, x - 1);
            bot_mid = referenceGray.at<uchar>(y + 1, x);
            bot_right = referenceGray.at<uchar>(y + 1, x + 1);

            gx = (top_right - top_left) +
                           2 * (mid_right - mid_left) +
                           (bot_right - bot_left);
            gy = (bot_left + 2 * bot_mid + bot_right) -
                           (top_left + 2 * top_mid + top_right);

            mag = std::abs(gx) + std::abs(gy);
            if (mag > 255) mag = 255;
            referenceSobel.at<uchar>(y - 1, x - 1) = static_cast<uchar>(mag);
        }
    }

    accumulateError(sobel_approx, referenceSobel, diff, errorStats);


    // Output statistics
    if (iterations > 1) {
        const double meanAbsoluteError = errorStats.pixels
            ? errorStats.sumAbsDiff / errorStats.pixels : 0.0;

        std::cout << std::fixed << std::setprecision(2);

        // Concise statistics
        std::cout << "Configuration: original\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Checksum: " << totalPixels << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Mean absolute error: " << meanAbsoluteError << "\n";
        std::cout << "Max difference: " << errorStats.maxDiff << "\n";
        std::cout << "Differing pixels: " << errorStats.differingPixels
                  << " / " << errorStats.pixels << "\n";

    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}