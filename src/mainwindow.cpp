#include "mainwindow.h"
#include "enginetaskwidget.h"

#include <QApplication>
#include <QDateTime>
#include <QFrame>
#include <QHBoxLayout>
#include <QScrollBar>
#include <QStatusBar>

// ─────────────────────────────────────────────────────────────────────────────
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("TRT Inference Studio");
    resize(1100, 720);
    setMinimumSize(800, 540);
    buildUi();
    refreshActionState();
}

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::buildUi() {
    // ── Global style ──────────────────────────────────────────────────────
    setStyleSheet(R"(
        QMainWindow, QWidget { background: #181825; color: #cdd6f4; }
        QScrollArea { border: none; background: transparent; }
        QScrollBar:vertical {
            background: #1e1e2e; width: 8px; border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #45475a; border-radius: 4px; min-height: 20px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        QSplitter::handle { background: #313244; width: 2px; }
        QTextEdit {
            background: #11111b;
            color: #a6adc8;
            border: 1px solid #313244;
            border-radius: 6px;
            font-family: "Consolas", "Courier New", monospace;
            font-size: 12px;
            padding: 6px;
        }
        QPushButton#primaryBtn {
            background: #89b4fa;
            color: #1e1e2e;
            border: none;
            border-radius: 5px;
            padding: 7px 20px;
            font-weight: 700;
        }
        QPushButton#primaryBtn:hover   { background: #74c7ec; }
        QPushButton#primaryBtn:disabled{ background: #313244; color:#585b70; }
        QPushButton#dangerBtn {
            background: #f38ba8;
            color: #1e1e2e;
            border: none;
            border-radius: 5px;
            padding: 7px 20px;
            font-weight: 700;
        }
        QPushButton#dangerBtn:disabled{ background:#313244; color:#585b70; }
        QPushButton#secondaryBtn {
            background: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 5px;
            padding: 6px 18px;
        }
        QPushButton#secondaryBtn:hover { background: #45475a; }
        QLabel#title {
            font-size: 18px;
            font-weight: 700;
            color: #89b4fa;
        }
        QLabel#subtitle { color:#6c7086; font-size: 12px; }
        QStatusBar { color:#585b70; font-size:11px; }
    )");

    // ── Central splitter ──────────────────────────────────────────────────
    splitter_ = new QSplitter(Qt::Horizontal, this);
    splitter_->setHandleWidth(2);
    setCentralWidget(splitter_);

    // ════════════════════════════════════════════════════════════════════════
    // LEFT PANEL
    // ════════════════════════════════════════════════════════════════════════
    auto* leftPanel = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setContentsMargins(16, 16, 8, 16);
    leftLayout->setSpacing(10);

    // Title
    auto* titleRow = new QHBoxLayout;
    auto* titleLbl = new QLabel("TRT Inference Studio");
    titleLbl->setObjectName("title");
    titleRow->addWidget(titleLbl);
    titleRow->addStretch();

    auto* subLbl = new QLabel("TensorRT 多引擎批量推理");
    subLbl->setObjectName("subtitle");
    titleRow->addWidget(subLbl);
    leftLayout->addLayout(titleRow);

    auto* sep = new QFrame;
    sep->setFrameShape(QFrame::HLine);
    sep->setStyleSheet("color:#313244; margin-bottom:4px;");
    leftLayout->addWidget(sep);

    // Toolbar
    auto* toolbar = new QHBoxLayout;
    addTaskBtn_ = new QPushButton("＋  添加任务");
    addTaskBtn_->setObjectName("primaryBtn");
    toolbar->addWidget(addTaskBtn_);

    startAllBtn_ = new QPushButton("▶▶  全部开始");
    startAllBtn_->setObjectName("primaryBtn");
    toolbar->addWidget(startAllBtn_);

    stopAllBtn_ = new QPushButton("■  全部停止");
    stopAllBtn_->setObjectName("dangerBtn");
    toolbar->addWidget(stopAllBtn_);

    toolbar->addStretch();
    leftLayout->addLayout(toolbar);

    // Global progress label
    globalProgress_ = new QLabel("就绪");
    globalProgress_->setStyleSheet("color:#a6adc8; font-size:12px;");
    leftLayout->addWidget(globalProgress_);

    // Scroll area holding task widgets
    scrollArea_ = new QScrollArea;
    scrollArea_->setWidgetResizable(true);
    scrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    taskContainer_ = new QWidget;
    taskContainer_->setStyleSheet("background:transparent;");
    taskLayout_ = new QVBoxLayout(taskContainer_);
    taskLayout_->setContentsMargins(0, 0, 6, 0);
    taskLayout_->setSpacing(10);
    taskLayout_->addStretch();        // keep tasks top-aligned

    scrollArea_->setWidget(taskContainer_);
    leftLayout->addWidget(scrollArea_, 1);

    splitter_->addWidget(leftPanel);

    // ════════════════════════════════════════════════════════════════════════
    // RIGHT PANEL – Log
    // ════════════════════════════════════════════════════════════════════════
    auto* rightPanel = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(8, 16, 16, 16);
    rightLayout->setSpacing(8);

    auto* logHeader = new QHBoxLayout;
    auto* logTitle = new QLabel("📋  运行日志");
    logTitle->setStyleSheet("font-weight:700; font-size:13px; color:#89b4fa;");
    logHeader->addWidget(logTitle);
    logHeader->addStretch();

    clearLogBtn_ = new QPushButton("清除");
    clearLogBtn_->setObjectName("secondaryBtn");
    clearLogBtn_->setFixedWidth(60);
    logHeader->addWidget(clearLogBtn_);
    rightLayout->addLayout(logHeader);

    logEdit_ = new QTextEdit;
    logEdit_->setReadOnly(true);
    logEdit_->setLineWrapMode(QTextEdit::WidgetWidth);
    rightLayout->addWidget(logEdit_, 1);

    splitter_->addWidget(rightPanel);
    splitter_->setStretchFactor(0, 3);
    splitter_->setStretchFactor(1, 2);

    // ── Status bar ────────────────────────────────────────────────────────
    statusBar()->showMessage("就绪  –  TRT Inference Studio  v1.0");

    // ── Connections ───────────────────────────────────────────────────────
    connect(addTaskBtn_,  &QPushButton::clicked, this, &MainWindow::addTask);
    connect(startAllBtn_, &QPushButton::clicked, this, &MainWindow::startAll);
    connect(stopAllBtn_,  &QPushButton::clicked, this, &MainWindow::stopAll);
    connect(clearLogBtn_, &QPushButton::clicked, this, &MainWindow::clearLog);

    // Seed one default task so the window isn't empty
    addTask();
}

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::addTask() {
    ++taskSerial_;
    auto* task = new EngineTaskWidget(taskSerial_, taskContainer_);

    connect(task, &EngineTaskWidget::logMessage,
            this, &MainWindow::appendLog);
    connect(task, &EngineTaskWidget::removeRequested,
            this, &MainWindow::removeTask);
    connect(task, &EngineTaskWidget::taskStarted,
            this, &MainWindow::onTaskStarted);
    connect(task, &EngineTaskWidget::taskFinished,
            this, &MainWindow::onTaskFinished);

    // Insert before the trailing stretch
    int insertPos = taskLayout_->count() - 1;
    taskLayout_->insertWidget(insertPos, task);
    tasks_.push_back(task);

    appendLog(QString("[系统] 已添加任务 #%1").arg(taskSerial_));
    refreshActionState();

    // Scroll to the new widget
    QTimer::singleShot(50, scrollArea_, [this]{
        scrollArea_->verticalScrollBar()->setValue(
            scrollArea_->verticalScrollBar()->maximum());
    });
}

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::removeTask(EngineTaskWidget* task) {
    auto it = std::find(tasks_.begin(), tasks_.end(), task);
    if (it == tasks_.end()) return;
    tasks_.erase(it);
    task->stopInference();
    taskLayout_->removeWidget(task);
    task->deleteLater();
    appendLog("[系统] 任务已移除");
    refreshActionState();
}

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::startAll() {
    for (auto* t : tasks_)
        if (t->isReady() && !t->isRunning())
            t->startInference();
}

void MainWindow::stopAll() {
    for (auto* t : tasks_)
        if (t->isRunning())
            t->stopInference();
}

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::onTaskStarted(EngineTaskWidget*) {
    ++runningCount_;
    updateGlobalProgress();
    refreshActionState();
}

void MainWindow::onTaskFinished(EngineTaskWidget*) {
    if (runningCount_ > 0) --runningCount_;
    updateGlobalProgress();
    refreshActionState();
}

void MainWindow::updateGlobalProgress() {
    if (runningCount_ == 0) {
        globalProgress_->setText("就绪");
        statusBar()->showMessage("所有任务完成  –  TRT Inference Studio  v1.0");
    } else {
        globalProgress_->setText(
            QString("正在运行：%1 个任务").arg(runningCount_));
        statusBar()->showMessage(
            QString("推理进行中  –  %1 个任务运行中").arg(runningCount_));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::refreshActionState() {
    bool anyReady   = std::any_of(tasks_.begin(), tasks_.end(),
                        [](auto* t){ return t->isReady() && !t->isRunning(); });
    bool anyRunning = runningCount_ > 0;

    startAllBtn_->setEnabled(anyReady);
    stopAllBtn_->setEnabled(anyRunning);
}

// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::appendLog(const QString& msg) {
    const QString ts = QDateTime::currentDateTime().toString("HH:mm:ss");
    logEdit_->append(
        QString("<span style='color:#585b70;'>%1</span> %2").arg(ts, msg));
    logEdit_->verticalScrollBar()->setValue(
        logEdit_->verticalScrollBar()->maximum());
}

void MainWindow::clearLog() {
    logEdit_->clear();
}
