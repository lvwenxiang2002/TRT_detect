#pragma once
#include <QObject>
#include <QRunnable>
#include <QString>
#include <QStringList>
#include <QAtomicInt>
#include <QMutex>
#include <memory>
#include <opencv2/core.hpp>

#include "trtengine.h"

// 🌟 移到此处：使得 saveResults 能返回检测结果
// 🌟 升级版：加入亚像素精度记忆
struct Detection {
    int classId;
    float conf;
    cv::Rect box;
    float exact_w; // 记录原始浮点型宽度
    float exact_h; // 记录原始浮点型高度
};

class ImageQueue {
public:
    explicit ImageQueue(QStringList paths);
    QString dequeue();
    int     remaining() const;
    int     total()     const { return total_; }
private:
    mutable QMutex  mutex_;
    QStringList     paths_;
    int             total_;
};

class InferenceWorker : public QObject, public QRunnable {
    Q_OBJECT
public:
    InferenceWorker(TRTEngine* engine, std::shared_ptr<ImageQueue> queue, const QString& outputDir,
        double resizeRatioW, double resizeRatioH, double pixelPrecision, bool drawLargeImg, QObject* parent = nullptr);

    void run() override;
    void requestStop() { stopRequested_ = true; }

signals:
    void imageFinished(const QString& imagePath, bool success);
    void workerDone();
    void errorOccurred(const QString& msg);

private:
    std::vector<float> preprocessImage(const cv::Mat& img);
    // 🌟 修改：返回检测结果以便大图映射
    std::vector<Detection> saveResults(const cv::Mat& cropImg, const QString& baseName, int x, int y, const QString& ext, const std::vector<float>& output);

    TRTEngine* engine_;
    std::shared_ptr<ImageQueue>  queue_;
    QString                      outputDir_;

    double resizeRatioW_;
    double resizeRatioH_;
    double pixelPrecision_;
    bool drawLargeImg_; // 🌟 大图映射开关

    std::atomic<bool>            stopRequested_{ false };
};