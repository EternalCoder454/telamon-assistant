// Starts Qt, makes the app single-instance and loads the window and the
// glow. All app logic is in Rust (src/); this file only glues.
#include <KDBusService>
#include <KWindowSystem>

#include <QApplication>
#include <QCommandLineParser>
#include <QPointer>
#include <QQmlApplicationEngine>
#include <QQmlEngine>
#include <QQuickWindow>
#include <QSGRendererInterface>

// Rust, see src/lib.rs.
extern "C" void *telamon_assistant_new();
extern "C" bool telamon_assistant_enabled();
// telamon-framework-ui (include/telamon/app.h), linked in with the Rust library.
extern "C" void telamon_app_init();
extern "C" void telamon_app_ready();

int main(int argc, char *argv[])
{
    // The journal logger, the crash hooks, the app ID, the desktop file
    // name, the version and the org.kde.desktop style.
    telamon_app_init();

    // The window and the glow are gradients and text: Qt Quick's software
    // backend draws them without loading Mesa. QT_QUICK_BACKEND overrides.
    if (qEnvironmentVariableIsEmpty("QT_QUICK_BACKEND")) {
        QQuickWindow::setGraphicsApi(QSGRendererInterface::Software);
    }

    QApplication app(argc, argv);
    telamon_app_ready();

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();
    const QCommandLineOption background(QStringLiteral("background"),
        QStringLiteral("Start without a window (at login); quit at once when Telamon is off."));
    parser.addOption(background);
    parser.process(app);

    // At login: nothing to do unless the user turned Telamon on.
    if (parser.isSet(background) && !telamon_assistant_enabled()) {
        return 0;
    }

    // One instance per session: a second launch shows this one's window.
    KDBusService service(KDBusService::Unique);
    // Closing the window keeps Telamon listening; Main.qml quits when it is off.
    app.setQuitOnLastWindowClosed(false);

    auto *assistant = static_cast<QObject *>(telamon_assistant_new());
    QQmlEngine::setObjectOwnership(assistant, QQmlEngine::CppOwnership);

    int rc = 0;
    {
        QQmlApplicationEngine engine;
        engine.setInitialProperties({
            {QStringLiteral("assistant"), QVariant::fromValue(assistant)},
            {QStringLiteral("background"), parser.isSet(background)},
        });
        QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app, [] { QCoreApplication::exit(1); }, Qt::QueuedConnection);
        engine.loadFromModule(QStringLiteral("net.eterneon.telamon.assistant"), QStringLiteral("Main"));

        QPointer<QQuickWindow> window = qobject_cast<QQuickWindow *>(engine.rootObjects().value(0));
        if (window) {
            QObject::connect(&service, &KDBusService::activateRequested, window, [window](const QStringList &, const QString &) {
                KWindowSystem::updateStartupId(window);
                window->show();
                window->raise();
                KWindowSystem::activateWindow(window);
            });
            rc = app.exec();
        } else {
            rc = 1;
        }
    }
    delete assistant;
    return rc;
}
