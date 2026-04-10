#pragma once
#include <QWidget>
#include <QThreadPool>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QSpinBox>
#include <QDoubleSpinBox> 
#include <QCheckBox> // 🌟 新增复选框
#include <QMutex>
#include <memory>
#include <vector>

#include "trtengine.h"
#include "inferenceworker.h"

class EngineTaskWidget : public QWidget {
    Q_OBJECT
public:
    explicit EngineTaskWidget(int index, QWidget* parent = nullptr);
    ~EngineTaskWidget() override;

    bool isReady()   const;
    bool isRunning() const { return running_; }

public slots:
    void startInference();
    void stopInference();

signals:
    void logMessage(const QString& msg);
    void removeRequested(EngineTaskWidget* self);
    void taskStarted(EngineTaskWidget* self);
    void taskFinished(EngineTaskWidget* self);

private slots:
    void onBrowseEngine();
    void onBrowseFolder();
    void onBrowseOutFolder();
    void onImageFinished(const QString& path, bool ok);
    void onWorkerDone();
    void onEngineLoadError(const QString& err);

private:
    void buildUi(int index);
    void updateStatus(const QString& text, const QString& color = {});
    QStringList collectImages() const;

    // ── UI widgets ────────────────────────────────────────────────────────
    QLabel* indexLabel_ = nullptr;
    QLineEdit* engineEdit_ = nullptr;
    QPushButton* engineBtn_ = nullptr;
    QLineEdit* folderEdit_ = nullptr;
    QPushButton* folderBtn_ = nullptr;
    QLineEdit* outFolderEdit_ = nullptr;
    QPushButton* outFolderBtn_ = nullptr;

    QSpinBox* cropWSpin_ = nullptr;
    QSpinBox* cropHSpin_ = nullptr;
    QSpinBox* overlapSpin_ = nullptr;
    QSpinBox* padSpin_ = nullptr;
    QDoubleSpinBox* pixelPrecisionSpin_ = nullptr;
    QSpinBox* threadSpin_ = nullptr;
    QCheckBox* drawLargeImgCheck_ = nullptr; // 🌟 新增：大图映射开关

    QProgressBar* progressBar_ = nullptr;
    QLabel* progressLabel_ = nullptr;
    QPushButton* startBtn_ = nullptr;
    QPushButton* stopBtn_ = nullptr;
    QPushButton* removeBtn_ = nullptr;
    QLabel* statusLabel_ = nullptr;

    // ── Inference state ───────────────────────────────────────────────────
    std::unique_ptr<TRTEngine>              engine_;
    std::unique_ptr<QThreadPool>            pool_;
    std::vector<InferenceWorker*>           workers_;
    std::shared_ptr<ImageQueue>             queue_;

    QString currentTaskFolder_;
    bool    running_ = false;
    int     totalImages_ = 0;
    int     doneImages_ = 0;
    int     activeWorkers_ = 0;
    QMutex  counterMutex_;
};