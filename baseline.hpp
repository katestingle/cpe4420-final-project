#include <thread>
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

    // processSegment is called concurrently, so the ring must not be shared
    // between workers.  Each thread gets one reusable three-row ring.
    thread_local std::array<cv::Mat, 3> grayRing;

    void ensureGrayRingWidth(int width) {
        for (cv::Mat& row : grayRing) {
            if (row.rows != 1 || row.cols != width || row.type() != CV_8UC1) {
                row.create(1, width, CV_8UC1);
            }
        }
    }

    void processSegment(const cv::Mat& frame, cv::Mat& sobelOutput,
                 int firstOutputRow, int lastOutputRow) {
        const int width = frame.cols;
        ensureGrayRingWidth(width);

        // Sobel output y needs grayscale input rows y, y + 1, and y + 2.
        for (int sourceY = firstOutputRow; sourceY < lastOutputRow + 2; ++sourceY) {
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
