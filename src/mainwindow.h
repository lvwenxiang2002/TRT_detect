#pragma once
#include <QMainWindow>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QSplitter>
#include <QTimer>

#include <vector>

class EngineTaskWidget;

// ─────────────────────────────────────────────────────────────────────────────
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void addTask();
    void removeTask(EngineTaskWidget* task);
    void startAll();
    void stopAll();
    void clearLog();
    void appendLog(const QString& msg);
    void onTaskStarted(EngineTaskWidget*);
    void onTaskFinished(EngineTaskWidget*);
    void updateGlobalProgress();

private:
    void buildUi();
    void refreshActionState();

    // ── Layout ────────────────────────────────────────────────────────────
    QWidget*     centralWidget_   = nullptr;
    QSplitter*   splitter_        = nullptr;

    // Left panel – task list
    QScrollArea* scrollArea_      = nullptr;
    QWidget*     taskContainer_   = nullptr;
    QVBoxLayout* taskLayout_      = nullptr;
    QPushButton* addTaskBtn_      = nullptr;
    QPushButton* startAllBtn_     = nullptr;
    QPushButton* stopAllBtn_      = nullptr;
    QLabel*      globalProgress_  = nullptr;

    // Right panel – log
    QTextEdit*   logEdit_         = nullptr;
    QPushButton* clearLogBtn_     = nullptr;

    // ── State ─────────────────────────────────────────────────────────────
    std::vector<EngineTaskWidget*> tasks_;
    int runningCount_ = 0;
    int taskSerial_   = 0;
};
