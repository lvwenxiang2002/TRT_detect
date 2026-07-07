#include "trtengine.h"
#include <QFile>
#include <QDebug>
#include <iostream>

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        qDebug() << "[TRT]" << msg;
    }
}

TRTEngine::TRTEngine(QObject* parent) : QObject(parent) {}
TRTEngine::~TRTEngine() { unload(); }

void TRTEngine::unload() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

bool TRTEngine::load(const QString& enginePath) {
    unload();
    enginePath_ = enginePath;

    QFile file(enginePath);
    if (!file.open(QIODevice::ReadOnly)) {
        lastError_ = "Failed to open engine file: " + enginePath;
        return false;
    }
    QByteArray engineData = file.readAll();
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        lastError_ = "Failed to create TRT runtime";
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.constData(), engineData.size()));
    if (!engine_) {
        lastError_ = "Failed to deserialize CUDA engine";
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        lastError_ = "Failed to create execution context";
        return false;
    }

    introspectBindings();
    return true;
}

void TRTEngine::introspectBindings() {
    int numBindings = engine_->getNbBindings();
    for (int i = 0; i < numBindings; ++i) {
        auto dims = engine_->getBindingDimensions(i);
        bool isInput = engine_->bindingIsInput(i);
        if (isInput) {
            inputBindingIdx_ = i;
            if (dims.nbDims >= 4) {
                inputChannels_ = dims.d[1];
                inputSize_ = QSize(dims.d[3], dims.d[2]);
            }
        } else {
            outputBindingIdx_ = i;
            outputSize_ = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                // Ignore dynamic batch size -1 if present
                if (dims.d[j] > 0) {
                    outputSize_ *= dims.d[j];
                }
            }
        }
    }
}

bool TRTEngine::infer(const std::vector<float>& inputBlob, std::vector<float>& outputBlob) {
    std::lock_guard<std::mutex> lock(inferMutex_);

    if (!context_ || inputBindingIdx_ < 0 || outputBindingIdx_ < 0) {
        lastError_ = "Engine not loaded or invalid bindings";
        return false;
    }

    outputBlob.resize(outputSize_);

    void* buffers[2] = {nullptr, nullptr};
    size_t inputSize = inputBlob.size() * sizeof(float);
    size_t outputSize = outputBlob.size() * sizeof(float);

    if (cudaMalloc(&buffers[inputBindingIdx_], inputSize) != cudaSuccess) return false;
    if (cudaMalloc(&buffers[outputBindingIdx_], outputSize) != cudaSuccess) {
        cudaFree(buffers[inputBindingIdx_]);
        return false;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        cudaFree(buffers[inputBindingIdx_]);
        cudaFree(buffers[outputBindingIdx_]);
        return false;
    }

    cudaMemcpyAsync(buffers[inputBindingIdx_], inputBlob.data(), inputSize, cudaMemcpyHostToDevice, stream);
    bool status = context_->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(outputBlob.data(), buffers[outputBindingIdx_], outputSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputBindingIdx_]);
    cudaFree(buffers[outputBindingIdx_]);

    return status;
}
