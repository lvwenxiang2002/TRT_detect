#include "enginetaskwidget.h"
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QMutexLocker>

static const QStringList kImageExts{ "*.jpg","*.jpeg","*.png","*.bmp","*.tiff","*.tif","*.webp" };

EngineTaskWidget::EngineTaskWidget(int index, QWidget* parent) : QWidget(parent) {
    engine_ = std::make_unique<TRTEngine>(this);
    buildUi(index);
}

EngineTaskWidget::~EngineTaskWidget() { stopInference(); }

void EngineTaskWidget::buildUi(int index) {
    setObjectName("EngineTaskWidget");
    setStyleSheet(R"(
        QWidget#EngineTaskWidget { background: #1e1e2e; border: 1px solid #313244; border-radius: 8px; }
        QLineEdit { background: #181825; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px 8px; }
        QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px 12px; }
        QPushButton:hover { background: #45475a; }
        QPushButton:pressed { background: #585b70; }
        QSpinBox, QDoubleSpinBox { background: #181825; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 2px 4px; }
        QCheckBox { color: #cdd6f4; font-weight: bold; }
    )");

    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    QGridLayout* grid = new QGridLayout();

    indexLabel_ = new QLabel(QString("<b>任务 %1</b>").arg(index), this);
    engineEdit_ = new QLineEdit(this); engineEdit_->setPlaceholderText("请选择 .engine 模型...");
    engineBtn_ = new QPushButton("浏览模型", this); connect(engineBtn_, &QPushButton::clicked, this, &EngineTaskWidget::onBrowseEngine);
    folderEdit_ = new QLineEdit(this); folderEdit_->setPlaceholderText("请选择输入图片文件夹...");
    folderBtn_ = new QPushButton("输入目录", this); connect(folderBtn_, &QPushButton::clicked, this, &EngineTaskWidget::onBrowseFolder);
    outFolderEdit_ = new QLineEdit(this); outFolderEdit_->setPlaceholderText("请选择结果保存文件夹...");
    outFolderBtn_ = new QPushButton("输出目录", this); connect(outFolderBtn_, &QPushButton::clicked, this, &EngineTaskWidget::onBrowseOutFolder);
    statusLabel_ = new QLabel("就绪", this); statusLabel_->setAlignment(Qt::AlignCenter);

    grid->addWidget(indexLabel_, 0, 0); grid->addWidget(statusLabel_, 0, 2);
    grid->addWidget(new QLabel("模型:"), 1, 0); grid->addWidget(engineEdit_, 1, 1); grid->addWidget(engineBtn_, 1, 2);
    grid->addWidget(new QLabel("输入:"), 2, 0); grid->addWidget(folderEdit_, 2, 1); grid->addWidget(folderBtn_, 2, 2);
    grid->addWidget(new QLabel("输出:"), 3, 0); grid->addWidget(outFolderEdit_, 3, 1); grid->addWidget(outFolderBtn_, 3, 2);
    mainLayout->addLayout(grid);

    // ── 控制栏 1: Resize 倍率 ──
    QHBoxLayout* ctrlLayout1 = new QHBoxLayout();
    ctrlLayout1->addWidget(new QLabel("宽倍率:")); resizeRatioWSpin_ = new QDoubleSpinBox(this); resizeRatioWSpin_->setDecimals(3); resizeRatioWSpin_->setRange(0.001, 100.0); resizeRatioWSpin_->setValue(1.0); ctrlLayout1->addWidget(resizeRatioWSpin_);
    ctrlLayout1->addWidget(new QLabel("高倍率:")); resizeRatioHSpin_ = new QDoubleSpinBox(this); resizeRatioHSpin_->setDecimals(3); resizeRatioHSpin_->setRange(0.001, 100.0); resizeRatioHSpin_->setValue(1.0); ctrlLayout1->addWidget(resizeRatioHSpin_);
    mainLayout->addLayout(ctrlLayout1);

    // ── 控制栏 2: 精度、线程数与大图开关 ──
    QHBoxLayout* ctrlLayout2 = new QHBoxLayout();
    ctrlLayout2->addWidget(new QLabel("精度(mm/px):"));
    pixelPrecisionSpin_ = new QDoubleSpinBox(this); pixelPrecisionSpin_->setDecimals(4); pixelPrecisionSpin_->setRange(0.0000, 100.0); pixelPrecisionSpin_->setValue(0.0100);
    ctrlLayout2->addWidget(pixelPrecisionSpin_);

    ctrlLayout2->addWidget(new QLabel("线程:")); threadSpin_ = new QSpinBox(this); threadSpin_->setRange(1, 32); threadSpin_->setValue(4); ctrlLayout2->addWidget(threadSpin_);

    // 🌟 新增：大图映射开关
    drawLargeImgCheck_ = new QCheckBox("开启原大图画框映射", this);
    drawLargeImgCheck_->setChecked(true); // 默认开启
    ctrlLayout2->addWidget(drawLargeImgCheck_);

    ctrlLayout2->addStretch();
    mainLayout->addLayout(ctrlLayout2);

    QHBoxLayout* progLayout = new QHBoxLayout();
    progressBar_ = new QProgressBar(this); progressBar_->setTextVisible(false); progLayout->addWidget(progressBar_);
    progressLabel_ = new QLabel("0 / 0", this); progLayout->addWidget(progressLabel_);

    startBtn_ = new QPushButton("开始", this);
    stopBtn_ = new QPushButton("暂停", this); stopBtn_->setEnabled(false);
    removeBtn_ = new QPushButton("移除任务", this);

    connect(startBtn_, &QPushButton::clicked, this, &EngineTaskWidget::startInference);
    connect(stopBtn_, &QPushButton::clicked, this, &EngineTaskWidget::stopInference);
    connect(removeBtn_, &QPushButton::clicked, [this]() { emit removeRequested(this); });

    progLayout->addWidget(startBtn_); progLayout->addWidget(stopBtn_); progLayout->addWidget(removeBtn_);
    mainLayout->addLayout(progLayout);
}

void EngineTaskWidget::onBrowseEngine() {
    QString path = QFileDialog::getOpenFileName(this, "选择 TensorRT Engine", "", "Engine Files (*.engine *.plan);;All Files (*.*)");
    if (!path.isEmpty()) engineEdit_->setText(path);
}
void EngineTaskWidget::onBrowseFolder() {
    QString dir = QFileDialog::getExistingDirectory(this, "选择输入图片文件夹");
    if (!dir.isEmpty()) folderEdit_->setText(dir);
}
void EngineTaskWidget::onBrowseOutFolder() {
    QString dir = QFileDialog::getExistingDirectory(this, "选择结果保存文件夹");
    if (!dir.isEmpty()) outFolderEdit_->setText(dir);
}
bool EngineTaskWidget::isReady() const { return !engineEdit_->text().isEmpty() && !folderEdit_->text().isEmpty() && !outFolderEdit_->text().isEmpty(); }
QStringList EngineTaskWidget::collectImages() const {
    QDir dir(folderEdit_->text());
    if (!dir.exists()) return {};
    QStringList files;
    for (const auto& fi : dir.entryInfoList(kImageExts, QDir::Files)) files << fi.absoluteFilePath();
    return files;
}

void EngineTaskWidget::startInference() {
    if (running_) return;

    QString enginePath = engineEdit_->text();
    QString inFolder = folderEdit_->text();
    QString outFolder = outFolderEdit_->text();

    if (enginePath.isEmpty() || inFolder.isEmpty() || outFolder.isEmpty()) { emit logMessage("启动失败: 请确保已选择所有路径！"); return; }
    if (!engine_->isLoaded()) { if (!engine_->load(enginePath)) { emit logMessage("模型加载失败: " + engine_->lastError()); return; } }

    bool isResume = false;
    if (queue_ && queue_->remaining() > 0 && currentTaskFolder_ == inFolder && doneImages_ < totalImages_) {
        isResume = true;
    }

    if (!isResume) {
        QStringList images = collectImages();
        if (images.isEmpty()) { emit logMessage("输入文件夹中未找到图片！"); return; }
        totalImages_ = images.size(); doneImages_ = 0;
        progressBar_->setMaximum(totalImages_); progressBar_->setValue(0);
        progressLabel_->setText(QString("0 / %1").arg(totalImages_));
        queue_ = std::make_shared<ImageQueue>(images);
        currentTaskFolder_ = inFolder;
    }
    else {
        emit logMessage(QString("▶ 从断点继续推理，剩余 %1 张...").arg(queue_->remaining()));
    }

    int threadCount = threadSpin_->value();
    activeWorkers_ = threadCount;
    running_ = true;

    if (!pool_) pool_ = std::make_unique<QThreadPool>();
    pool_->setMaxThreadCount(threadCount);

    double resizeRatioW = resizeRatioWSpin_->value();
    double resizeRatioH = resizeRatioHSpin_->value();
    double precision = pixelPrecisionSpin_->value();
    bool drawLarge = drawLargeImgCheck_->isChecked(); // 🌟 提取大图映射开关状态

    for (int i = 0; i < threadCount; ++i) {
        auto* worker = new InferenceWorker(engine_.get(), queue_, outFolder, resizeRatioW, resizeRatioH, precision, drawLarge, this);
        connect(worker, &InferenceWorker::imageFinished, this, &EngineTaskWidget::onImageFinished, Qt::QueuedConnection);
        connect(worker, &InferenceWorker::workerDone, this, &EngineTaskWidget::onWorkerDone, Qt::QueuedConnection);
        connect(worker, &InferenceWorker::errorOccurred, this, &EngineTaskWidget::onEngineLoadError, Qt::QueuedConnection);
        workers_.push_back(worker);
        pool_->start(worker);
    }

    updateStatus("运行中...", "#f9e2af");
    startBtn_->setEnabled(false); stopBtn_->setEnabled(true);
    emit taskStarted(this);
}

void EngineTaskWidget::stopInference() {
    if (!running_) return;
    for (auto* w : workers_) if (w) w->requestStop();
    updateStatus("正在暂停...", "#f38ba8"); stopBtn_->setEnabled(false);
}

void EngineTaskWidget::onImageFinished(const QString& path, bool ok) {
    QMutexLocker lk(&counterMutex_);
    ++doneImages_;
    progressBar_->setValue(doneImages_);
    progressLabel_->setText(QString("%1 / %2").arg(doneImages_).arg(totalImages_));
    if (!ok) emit logMessage(QString("  ✗ 推理异常: %1").arg(QFileInfo(path).fileName()));
}

void EngineTaskWidget::onWorkerDone() {
    QMutexLocker lk(&counterMutex_);
    --activeWorkers_;
    if (activeWorkers_ > 0) return;

    running_ = false;
    for (auto* w : workers_) w->deleteLater();
    workers_.clear();

    if (doneImages_ >= totalImages_) {
        updateStatus("完成 ✓", "#a6e3a1");
        emit logMessage(QString("[Task %1] 任务彻底完成！正在清理释放显存...").arg(indexLabel_->text().remove("<b>").remove("</b>")));

        // 🌟 核心修复：彻底销毁当前加载了模型的引擎实例（触发底层的 cudaFree），释放显存！
        // 然后立刻 new 一个新的空壳引擎，这样如果你再次点击“开始”，它会自动重新走读取硬盘加载模型的流程。
        engine_.reset(new TRTEngine(this));

    }
    else {
        // 如果只是中途暂停，绝对不要释放显存，否则继续时会卡顿
        updateStatus("已暂停 (可继续)", "#fab387");
    }

    startBtn_->setEnabled(true); stopBtn_->setEnabled(false);
    emit taskFinished(this);
}

void EngineTaskWidget::onEngineLoadError(const QString& err) { emit logMessage(err); }
void EngineTaskWidget::updateStatus(const QString& text, const QString& color) {
    statusLabel_->setText(text);
    if (!color.isEmpty()) statusLabel_->setStyleSheet(QString("color:%1; font-weight:bold; background:#1e1e2e; border:none;").arg(color));
}