#pragma once
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <QObject>
#include <QString>
#include <QSize>

#include <memory>
#include <mutex>
#include <vector>

// ── TRT 8.5 自定义删除器 ──────────────────────────────────────────────────────
struct TRTDestroyer {
    template<typename T>
    void operator()(T* p) const noexcept { if (p) p->destroy(); }
};

template<typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroyer>;

// ── Logger ────────────────────────────────────────────────────────────────────
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// ── TRTEngine ─────────────────────────────────────────────────────────────────
class TRTEngine : public QObject {
    Q_OBJECT
public:
    explicit TRTEngine(QObject* parent = nullptr);
    ~TRTEngine() override;

    bool    load(const QString& enginePath);
    void    unload();
    bool    isLoaded()      const { return engine_ != nullptr; }

    QSize   inputSize()     const { return inputSize_; }
    int     inputChannels() const { return inputChannels_; }
    int     numClasses()    const { return numClasses_; }
    QString lastError()     const { return lastError_; }
    QString enginePath()    const { return enginePath_; }

    // 🌟 核心：只有一个参数极其干净的 infer 函数，内部自己管锁
    bool infer(const std::vector<float>& inputBlob,
        std::vector<float>& outputBlob);

private:
    void introspectBindings();

    QString  enginePath_;
    QString  lastError_;

    TRTLogger                           logger_;
    TRTUniquePtr<nvinfer1::IRuntime>    runtime_;
    TRTUniquePtr<nvinfer1::ICudaEngine> engine_;

    // 🌟 核心：唯一的执行上下文和排队锁
    TRTUniquePtr<nvinfer1::IExecutionContext> context_;
    std::mutex                                inferMutex_;

    QSize inputSize_{ 640, 640 };
    int   inputChannels_ = 3;
    int   inputBindingIdx_ = -1;
    int   outputBindingIdx_ = -1;
    int   outputSize_ = 0;
    int   numClasses_ = 0;
};