#include "mainwindow.h"
#include <QApplication>
#include <QIcon>

int main(int argc, char* argv[])
{
    // 🌟 核心修复：开启 Qt5 的高分屏(High-DPI)支持！
    // ⚠️ 极其重要：这几行代码必须写在 QApplication app(...) 之前！
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

    // 让非整数缩放（如 150%）更平滑，防止控件边缘出现奇怪的锯齿或白边
    qputenv("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough");
#endif

    // 原有的应用实例创建
    QApplication app(argc, argv);
    app.setApplicationName("TRT Inference Studio");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("IndustrialVision");

    // Optional: embed an icon via Qt resource system
    // app.setWindowIcon(QIcon(":/icons/app.png"));

    MainWindow w;
    w.show();

    return app.exec();
}