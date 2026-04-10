#include "inferenceworker.h"

#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QMutexLocker>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// ImageQueue 实现
// ─────────────────────────────────────────────────────────────────────────────
ImageQueue::ImageQueue(QStringList paths) : paths_(std::move(paths)), total_(paths_.size()) {}
QString ImageQueue::dequeue() { QMutexLocker lk(&mutex_); if (paths_.isEmpty()) return {}; return paths_.takeFirst(); }
int ImageQueue::remaining() const { QMutexLocker lk(&mutex_); return paths_.size(); }

// ─────────────────────────────────────────────────────────────────────────────
// InferenceWorker 实现
// ─────────────────────────────────────────────────────────────────────────────
InferenceWorker::InferenceWorker(TRTEngine* engine, std::shared_ptr<ImageQueue> queue, const QString& outputDir,
    int cropW, int cropH, int overlap, int padVal, double pixelPrecision, bool drawLargeImg, QObject* parent)
    : QObject(parent), QRunnable(), engine_(engine), queue_(std::move(queue)), outputDir_(outputDir),
    cropW_(cropW), cropH_(cropH), overlap_(overlap), padVal_(padVal), pixelPrecision_(pixelPrecision), drawLargeImg_(drawLargeImg)
{
    setAutoDelete(false);
}

void InferenceWorker::run() {
    try {
        while (!stopRequested_) {
            QString imgPath = queue_->dequeue();
            if (imgPath.isEmpty()) break;

            cv::Mat largeImg = cv::imread(imgPath.toLocal8Bit().constData(), cv::IMREAD_COLOR);
            if (largeImg.empty()) {
                emit errorOccurred(QString("无法读取图片: %1").arg(imgPath));
                emit imageFinished(imgPath, false);
                continue;
            }

            int strideX = cropW_ - overlap_;
            int strideY = cropH_ - overlap_;
            if (strideX <= 0) strideX = cropW_;
            if (strideY <= 0) strideY = cropH_;

            QFileInfo fi(imgPath);
            QString baseName = fi.completeBaseName();
            QString ext = fi.suffix();
            bool allCropsOk = true;

            // 🌟 容器：收集整张大图上的所有缺陷框
            std::vector<Detection> globalDets;

            for (int y = 0; y < largeImg.rows; y += strideY) {
                for (int x = 0; x < largeImg.cols; x += strideX) {
                    if (stopRequested_) break;

                    int rectW = std::min(cropW_, largeImg.cols - x);
                    int rectH = std::min(cropH_, largeImg.rows - y);
                    cv::Rect roi(x, y, rectW, rectH);

                    cv::Mat crop = largeImg(roi).clone();

                    if (rectW < cropW_ || rectH < cropH_) {
                        cv::copyMakeBorder(crop, crop, 0, cropH_ - rectH, 0, cropW_ - rectW,
                            cv::BORDER_CONSTANT, cv::Scalar(padVal_, padVal_, padVal_));
                    }

                    std::vector<float> inputBlob = preprocessImage(crop);
                    if (inputBlob.empty()) continue;

                    std::vector<float> outputBlob;
                    bool ok = engine_->infer(inputBlob, outputBlob);
                    if (!ok) { allCropsOk = false; continue; }

                    if (ok && !outputDir_.isEmpty()) {
                        // 🌟 保存小图，并接收该小图返回的高精度坐标框
                        auto cropDets = saveResults(crop, baseName, x, y, ext, outputBlob);
                        for (auto& d : cropDets) {
                            // 映射回大图上的绝对像素坐标
                            d.box.x += x;
                            d.box.y += y;
                            globalDets.push_back(d);
                        }
                    }
                }
            }

            // 🌟 核心：大图映射保存逻辑
            if (drawLargeImg_ && !globalDets.empty() && !outputDir_.isEmpty() && !stopRequested_) {
                // 1. 全局 NMS 去重 (解决滑动窗口的 overlap 重复检测问题)
                std::vector<cv::Rect> bxs;
                std::vector<float> cfs;
                for (const auto& d : globalDets) {
                    bxs.push_back(d.box);
                    cfs.push_back(d.conf);
                }
                std::vector<int> finalIdx;
                cv::dnn::NMSBoxes(bxs, cfs, 0.40f, 0.45f, finalIdx);

                // 2. 在原大图上直接绘制 (采用防遮挡描边 + 亚像素面积计算)
                cv::Mat drawLarge = largeImg.clone();

                // 稍微收敛一下放大系数，防止 8K/16K 图上文字过于巨大
                double scale = std::max(1.0, drawLarge.cols / 4096.0);
                int thick = std::max(2, static_cast<int>(scale * 1.5));
                double fontScale = std::max(0.5, scale * 0.4);

                for (int i : finalIdx) {
                    const auto& d = globalDets[i];

                    // 画目标的红框
                    cv::rectangle(drawLarge, d.box, cv::Scalar(0, 0, 255), thick);

                    // ── 绘制顶部标签 (类别和置信度) ──
                    QString txt = QString("C:%1 P:%2").arg(d.classId).arg(d.conf, 0, 'f', 2);
                    int baseLine;
                    cv::Size sz = cv::getTextSize(txt.toStdString(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thick, &baseLine);

                    // 智能避让
                    int textY = d.box.y - 8;
                    if (textY - sz.height < 0) textY = d.box.y + d.box.height + sz.height + 8;

                    // 描边文字 (底层黑边，表层黄字)
                    cv::putText(drawLarge, txt.toStdString(), cv::Point(d.box.x, textY),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thick + 2);
                    cv::putText(drawLarge, txt.toStdString(), cv::Point(d.box.x, textY),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255), thick);

                    // ── 绘制底部标签 (面积和对角线) ──
                    if (pixelPrecision_ > 0.0) {
                        // 🌟 物理换算：使用绝对无损的亚像素尺寸 exact_w / exact_h
                        double w_mm = d.exact_w * pixelPrecision_;
                        double h_mm = d.exact_h * pixelPrecision_;
                        double area = w_mm * h_mm;
                        double diag = std::sqrt(w_mm * w_mm + h_mm * h_mm);

                        QString measureText = QString("S:%1 D:%2").arg(area, 0, 'f', 2).arg(diag, 0, 'f', 2);
                        int baseLine2;
                        cv::Size sz2 = cv::getTextSize(measureText.toStdString(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thick, &baseLine2);

                        // 智能避让
                        int measureY = d.box.y + d.box.height + sz2.height + 8;
                        if (textY > d.box.y) measureY = textY + sz2.height + 8;
                        if (measureY > drawLarge.rows) measureY = d.box.y + d.box.height - 8;

                        // 描边文字 (底层黑边，表层白字)
                        cv::putText(drawLarge, measureText.toStdString(), cv::Point(d.box.x, measureY),
                            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thick + 2);
                        cv::putText(drawLarge, measureText.toStdString(), cv::Point(d.box.x, measureY),
                            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thick);
                    }
                }

                // 3. 不压缩，按原始超大分辨率存盘
                QDir dir(outputDir_);
                dir.mkpath("drawn_large");
                QString largePath = dir.filePath("drawn_large/" + fi.completeBaseName() + "_mapped." + fi.suffix());
                cv::imwrite(largePath.toLocal8Bit().constData(), drawLarge);
            }

            emit imageFinished(imgPath, allCropsOk);
        }
    }
    catch (const cv::Exception& e) {
        emit errorOccurred(QString("OpenCV 异常: %1").arg(e.what()));
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("C++ 异常: %1").arg(e.what()));
    }
    catch (...) {
        emit errorOccurred("未知异常导致线程退出");
    }
    emit workerDone();
}

std::vector<float> InferenceWorker::preprocessImage(const cv::Mat& img) {
    const QSize sz = engine_->inputSize();
    const int   w = sz.width();
    const int   h = sz.height();
    const int   chan = engine_->inputChannels();

    cv::Mat resized;
    cv::resize(img, resized, { w, h });
    cv::Mat flt;
    resized.convertTo(flt, CV_32F, 1.0 / 255.0);

    std::vector<float> blob(chan * h * w);
    std::vector<cv::Mat> chans(chan);
    cv::split(flt, chans);
    if (chan == 3) std::swap(chans[0], chans[2]);
    for (int c = 0; c < chan; ++c)
        std::memcpy(blob.data() + c * h * w, chans[c].ptr<float>(), h * w * sizeof(float));
    return blob;
}

std::vector<Detection> InferenceWorker::saveResults(const cv::Mat& cropImg, const QString& baseName, int x, int y, const QString& ext, const std::vector<float>& output)
{
    float imgW = static_cast<float>(cropImg.cols);
    float imgH = static_cast<float>(cropImg.rows);
    int inW = engine_->inputSize().width();
    int inH = engine_->inputSize().height();
    float rx = imgW / static_cast<float>(inW);
    float ry = imgH / static_cast<float>(inH);

    // P5/P6 架构自动探测，彻底防错位
    int anchors_p5 = (inW / 8 * inH / 8) + (inW / 16 * inH / 16) + (inW / 32 * inH / 32);
    int anchors_p6 = anchors_p5 + (inW / 64 * inH / 64);
    int num_anchors = anchors_p5;
    if (output.size() % anchors_p6 == 0) num_anchors = anchors_p6;
    else if (output.size() % anchors_p5 == 0) num_anchors = anchors_p5;
    else num_anchors = static_cast<int>(output.size()) / 5;

    int num_classes = static_cast<int>(output.size()) / num_anchors - 4;
    if (num_classes < 1) num_classes = 1;

    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<float> exactWs;
    std::vector<float> exactHs;

    boxes.reserve(500); confidences.reserve(500); classIds.reserve(500);
    exactWs.reserve(500); exactHs.reserve(500);

    const float* outPtr = output.data();
    const float* classScores = outPtr + 4 * num_anchors;

    for (int i = 0; i < num_anchors; ++i) {
        float maxConf = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < num_classes; ++c) {
            float conf = classScores[c * num_anchors + i];
            if (conf > maxConf) { maxConf = conf; maxClassId = c; }
        }
        if (maxConf > confThreshold) {
            float cx = outPtr[0 * num_anchors + i];
            float cy = outPtr[1 * num_anchors + i];
            float w = outPtr[2 * num_anchors + i];
            float h = outPtr[3 * num_anchors + i];

            // 🌟 获取绝对无损的亚像素尺寸
            float exact_w = w * rx;
            float exact_h = h * ry;

            int left = static_cast<int>(cx * rx - exact_w * 0.5f);
            int top = static_cast<int>(cy * ry - exact_h * 0.5f);
            int width = static_cast<int>(exact_w);
            int height = static_cast<int>(exact_h);

            boxes.emplace_back(left, top, width, height);
            confidences.emplace_back(maxConf);
            classIds.emplace_back(maxClassId);
            exactWs.emplace_back(exact_w);
            exactHs.emplace_back(exact_h);
        }
    }

    if (boxes.empty()) return {};

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices, 1.0f, 300);

    std::vector<Detection> dets;
    dets.reserve(indices.size());
    for (int idx : indices) {
        dets.push_back({ classIds[idx], confidences[idx], boxes[idx], exactWs[idx], exactHs[idx] });
    }

    if (dets.empty()) return {};

    QString cropName = QString("%1_%2_%3.%4").arg(baseName).arg(x).arg(y).arg(ext);
    QString labelName = QString("%1_%2_%3.txt").arg(baseName).arg(x).arg(y);

    QDir dir(outputDir_);
    dir.mkpath("orig"); dir.mkpath("labels"); dir.mkpath("drawn");

    QString origPath = dir.filePath("orig/" + cropName);
    QString labelPath = dir.filePath("labels/" + labelName);
    QString drawnPath = dir.filePath("drawn/" + cropName);

    // 保存原图和小图标签
    cv::imwrite(origPath.toLocal8Bit().constData(), cropImg);

    QFile labelFile(labelPath);
    if (labelFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream ts(&labelFile);
        for (const auto& d : dets) {
            float cx = (d.box.x + d.box.width / 2.0f) / imgW;
            float cy = (d.box.y + d.box.height / 2.0f) / imgH;
            float norm_w = static_cast<float>(d.box.width) / imgW;
            float norm_h = static_cast<float>(d.box.height) / imgH;
            ts << d.classId << " " << QString::number(cx, 'f', 6) << " " << QString::number(cy, 'f', 6) << " "
                << QString::number(norm_w, 'f', 6) << " " << QString::number(norm_h, 'f', 6) << "\n";
        }
        labelFile.close();
    }

    // 在裁切小图上绘制防遮挡的悬浮文字
    cv::Mat drawImg = cropImg.clone();
    int thick = 2;
    double fontScale = 0.5;

    for (const auto& d : dets) {
        cv::rectangle(drawImg, d.box, cv::Scalar(0, 0, 255), thick);

        QString txt = QString("C:%1 P:%2").arg(d.classId).arg(d.conf, 0, 'f', 2);
        int baseLine;
        cv::Size sz = cv::getTextSize(txt.toStdString(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thick, &baseLine);

        int textY = d.box.y - 5;
        if (textY - sz.height < 0) textY = d.box.y + d.box.height + sz.height + 5;

        // 描边文字
        cv::putText(drawImg, txt.toStdString(), cv::Point(d.box.x, textY),
            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thick + 2);
        cv::putText(drawImg, txt.toStdString(), cv::Point(d.box.x, textY),
            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255), thick);

        if (pixelPrecision_ > 0.0) {
            double w_mm = d.exact_w * pixelPrecision_;
            double h_mm = d.exact_h * pixelPrecision_;
            double area = w_mm * h_mm;
            double diag = std::sqrt(w_mm * w_mm + h_mm * h_mm);

            QString measureText = QString("S:%1 D:%2").arg(area, 0, 'f', 2).arg(diag, 0, 'f', 2);
            int baseLine2;
            cv::Size sz2 = cv::getTextSize(measureText.toStdString(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thick, &baseLine2);

            int measureY = d.box.y + d.box.height + sz2.height + 5;
            if (textY > d.box.y) measureY = textY + sz2.height + 5;
            if (measureY > drawImg.rows) measureY = d.box.y + d.box.height - 5;

            // 描边文字
            cv::putText(drawImg, measureText.toStdString(), cv::Point(d.box.x, measureY),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thick + 2);
            cv::putText(drawImg, measureText.toStdString(), cv::Point(d.box.x, measureY),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thick);
        }
    }
    cv::imwrite(drawnPath.toLocal8Bit().constData(), drawImg);

    return dets;
}