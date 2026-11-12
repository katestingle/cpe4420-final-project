#include <algorithm>
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
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

namespace operations {

constexpr int kGrayShift = 8;
constexpr int kRedWeight = 54;    // round(0.2126 * 256)
constexpr int kGreenWeight = 183; // round(0.7152 * 256)
constexpr int kBlueWeight = 18;   // round(0.0722 * 256)
constexpr int kGrayRounding = 1 << (kGrayShift - 1);

// The main thread is also a processing participant. Only the remaining
// participants need background threads.
class RowWorkerPool {
public:
    explicit RowWorkerPool(int participantCount)
        : participantCount_(participantCount), workerCount_(participantCount - 1) {
        workers_.reserve(workerCount_);
        for (int workerId = 0; workerId < workerCount_; ++workerId) {
            workers_.emplace_back(&RowWorkerPool::workerLoop, this, workerId);
        }
    }

    RowWorkerPool(const RowWorkerPool&) = delete;
    RowWorkerPool& operator=(const RowWorkerPool&) = delete;

    ~RowWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        workReady_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    void runRows(int rowCount, std::function<void(int, int)> task) {
        if (workerCount_ == 0) {
            task(0, rowCount);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            rowCount_ = rowCount;
            completedWorkers_ = 0;
            ++generation_;
        }
        workReady_.notify_all();

        // The main thread handles the final segment while the workers run.
        runParticipant(task, rowCount, workerCount_);

        std::unique_lock<std::mutex> lock(mutex_);
        workDone_.wait(lock, [this] { return completedWorkers_ == workerCount_; });
    }

private:
    void runParticipant(const std::function<void(int, int)>& task, int rowCount,
                        int participantId) const {
        const int rowsPerParticipant =
            (rowCount + participantCount_ - 1) / participantCount_;
        const int firstRow = participantId * rowsPerParticipant;
        const int lastRow = std::min(rowCount, firstRow + rowsPerParticipant);
        if (firstRow < lastRow) task(firstRow, lastRow);
    }

    void workerLoop(int workerId) {
        int seenGeneration = 0;
        while (true) {
            std::function<void(int, int)> task;
            int rowCount = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                workReady_.wait(lock, [this, seenGeneration] {
                    return stopping_ || generation_ != seenGeneration;
                });
                if (stopping_) return;

                seenGeneration = generation_;
                task = task_;
                rowCount = rowCount_;
            }

            runParticipant(task, rowCount, workerId);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (++completedWorkers_ == workerCount_) workDone_.notify_one();
            }
        }
    }

    const int participantCount_;
    const int workerCount_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable workReady_;
    std::condition_variable workDone_;
    std::function<void(int, int)> task_;
    int rowCount_ = 0;
    int generation_ = 0;
    int completedWorkers_ = 0;
    bool stopping_ = false;
};

    void processSegment(const cv::Mat& frame, cv::Mat& sobelOutput,
                 int firstOutputRow, int lastOutputRow) {
    const int width = frame.cols;

    // Create a 3 row ring buffer for grayscale
    cv::Mat grayRing[3] = {
        cv::Mat(1, width, CV_8UC1),
        cv::Mat(1, width, CV_8UC1),
        cv::Mat(1, width, CV_8UC1)
    };

    // Sobel output y needs grayscale input rows y, y + 1, and y + 2.
    for (int sourceY = firstOutputRow;
         sourceY < lastOutputRow + 2;
         ++sourceY) {
        const uchar* src = frame.ptr<uchar>(sourceY);
        uchar* gray = grayRing[sourceY % 3].ptr<uchar>(0);

        // Convert row to grayscale
        for (int x = 0; x < width; ++x) {
            const int i = 3 * x;
            gray[x] = static_cast<uchar>(
                (kRedWeight * src[i + 2] +
                 kGreenWeight * src[i + 1] +
                 kBlueWeight * src[i] +
                 kGrayRounding) >> kGrayShift);
        }

        // Once three consecutive grayscale rows exist, produce the
        // Sobel row whose top input is sourceY - 2.
        if (sourceY < firstOutputRow + 2) {
            continue;
        }

        const int outputY = sourceY - 2;
        const uchar* top = grayRing[(sourceY - 2) % 3].ptr<uchar>(0);
        const uchar* middle = grayRing[(sourceY - 1) % 3].ptr<uchar>(0);
        const uchar* bottom = grayRing[sourceY % 3].ptr<uchar>(0);
        uchar* dst = sobelOutput.ptr<uchar>(outputY);

        for (int x = 1; x < width - 1; ++x) {
            const int gx = (top[x + 1] - top[x - 1]) +
                           2 * (middle[x + 1] - middle[x - 1]) +
                           (bottom[x + 1] - bottom[x - 1]);

            const int gy = (bottom[x - 1] + 2 * bottom[x] + bottom[x + 1]) -
                           (top[x - 1] + 2 * top[x] + top[x + 1]);

            dst[x - 1] = static_cast<uchar>(
                std::min(255, std::abs(gx) + std::abs(gy)));
        }
    }
}

struct ErrorStats {
    double sumAbsDiff = 0.0;
    uint64_t pixels = 0;
    uint64_t differingPixels = 0;
    int maxDiff = 0;
};

void accumulateError(const cv::Mat& test, const cv::Mat& reference,
                     cv::Mat& diff, ErrorStats& stats) {
    cv::absdiff(test, reference, diff);
    double frameMax = 0.0;
    cv::minMaxLoc(diff, nullptr, &frameMax);
    stats.maxDiff = std::max(stats.maxDiff, static_cast<int>(frameMax));
    stats.sumAbsDiff += cv::sum(diff)[0];
    stats.differingPixels += static_cast<uint64_t>(cv::countNonZero(diff));
    stats.pixels += static_cast<uint64_t>(diff.total());
}

} // namespace operations

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


    operations::RowWorkerPool workers(threadCount);
    cv::Mat refSobel, diff;
    operations::ErrorStats errorStats;
    uint64_t totalSobelPixels = 0;

    for (int i = 0; i < iterations; ++i) {
        if (frame.empty()) break;

        workers.runRows(rows - 2, [&](int firstRow, int lastRow) {
            operations::processSegment(frame, sobelOutput, firstRow, lastRow);
        });
        
        if (display) {
            cv::imshow("Output", sobelOutput);
            cv::waitKey(1);
        }

        totalSobelPixels += static_cast<uint64_t>(sobelOutput.total());
    }

    // Single-threaded reference and error (outside all timed sections)
    refSobel.create(rows - 2, cols - 2, CV_8UC1);
    operations::processSegment(frame, refSobel, 0, rows - 2);
    operations::accumulateError(sobelOutput, refSobel, diff, errorStats);

    // Output statistics
    if (iterations > 1) {
        const double meanError = errorStats.pixels
            ? errorStats.sumAbsDiff / errorStats.pixels : 0.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Configuration: optimized_" << threadCount << "T\n";
        std::cout << "Thread count: " << threadCount << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Checksum: " << totalSobelPixels << "\n";
        
        std::cout << std::setprecision(6);
        std::cout << "Mean absolute error: " << meanError << "\n";
        std::cout << "Max difference: " << errorStats.maxDiff << "\n";
        std::cout << "Differing pixels: " << errorStats.differingPixels
                  << " / " << errorStats.pixels << "\n";
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}