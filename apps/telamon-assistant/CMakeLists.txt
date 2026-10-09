cmake_minimum_required(VERSION 3.24)
project(telamon-assistant VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX /usr CACHE PATH "" FORCE)
endif()
include(GNUInstallDirs)
find_package(Qt6 6.11 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 Widgets)
qt_policy(SET QTP0001 NEW)
qt_policy(SET QTP0004 NEW)
# telamon-framework-ui's startup code (single instance, window activation)
find_package(KF6DBusAddons REQUIRED)
find_package(KF6WindowSystem REQUIRED)
find_package(Corrosion REQUIRED)
# The weather tool's HTTPS (voice-core, through ureq).
find_package(OpenSSL REQUIRED)
# whisper.cpp's GPU backend and CPU threads (built into the Rust library by
# whisper-rs; its link flags don't reach this link through Corrosion).
find_package(Vulkan REQUIRED)
find_package(OpenMP REQUIRED)

set(WORKSPACE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

find_program(QMAKE6 NAMES qmake6 qmake REQUIRED)
corrosion_import_crate(MANIFEST_PATH ${WORKSPACE_DIR}/Cargo.toml
    CRATES telamon-assistant LOCKED)
corrosion_set_env_vars(telamon_assistant QMAKE=${QMAKE6})

if(NOT EXISTS "${QT6_INSTALL_PREFIX}/${QT6_INSTALL_QML}/Telamon/Ui/qmldir")
    message(FATAL_ERROR "Telamon.Ui is not installed: install telamon-ui (Telamon framework)")
endif()

qt_add_executable(telamon-assistant-app cpp/main.cpp)
set_target_properties(telamon-assistant-app PROPERTIES OUTPUT_NAME telamon-assistant)

qt_add_qml_module(telamon-assistant-app
    URI net.eterneon.telamon.assistant
    VERSION 1.0
    QML_FILES
        qml/Main.qml
        qml/GlowWindow.qml
        qml/Glow.qml
)

# whisper.cpp (static, built by whisper-rs) and ONNX Runtime (loaded at run
# time) come with the Rust library.
target_link_libraries(telamon-assistant-app PRIVATE
    telamon_assistant
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::Widgets
    KF6::DBusAddons KF6::WindowSystem
    OpenSSL::SSL OpenSSL::Crypto
    Vulkan::Vulkan OpenMP::OpenMP_CXX)

install(TARGETS telamon-assistant-app RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES data/net.eterneon.telamon.assistant.desktop
        DESTINATION ${CMAKE_INSTALL_DATADIR}/applications)
install(FILES data/net.eterneon.telamon.assistant-autostart.desktop
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/xdg/autostart
        RENAME net.eterneon.telamon.assistant.desktop)
install(FILES data/net.eterneon.telamon.assistant.svg
        DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps)
