import QtQuick
import QtQuick.Window
import org.kde.layershell as LayerShell
import org.kde.kirigami as Kirigami
import Telamon.Ui

// The glow around one screen's edges, above every window, letting every
// click through. On Wayland it is a layer-shell overlay (KWin places it
// over the whole screen); on X11 a frameless tool window the screen's size.
// Not shown, and so not drawn, while Telamon sleeps.
Window {
    id: win

    property bool active: false
    property real level: 0
    property bool thinking: false
    property real pulse: 0
    // The glow's strength now: the voice, or the thinking pulse.
    readonly property real strength: Math.max(win.level, win.pulse)

    LayerShell.Window.scope: "telamon-glow"
    LayerShell.Window.layer: LayerShell.Window.LayerOverlay
    LayerShell.Window.anchors: LayerShell.Window.AnchorTop | LayerShell.Window.AnchorBottom | LayerShell.Window.AnchorLeft | LayerShell.Window.AnchorRight
    LayerShell.Window.exclusionZone: -1
    LayerShell.Window.keyboardInteractivity: LayerShell.Window.KeyboardInteractivityNone

    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus | Qt.Tool
    color: "transparent"
    title: qsTr("Telamon Glow")
    x: win.screen ? win.screen.virtualX : 0
    y: win.screen ? win.screen.virtualY : 0
    width: win.screen ? win.screen.width : 0
    height: win.screen ? win.screen.height : 0
    visible: win.active || fade.running

    readonly property color gold: "#F5C451"
    readonly property color bronze: "#B87333"
    // The strength, eased so the glow moves smoothly between level updates.
    property real eased: win.strength
    readonly property real reach: Kirigami.Units.gridUnit * (1.5 + 3.5 * win.eased)

    Behavior on eased {
        enabled: !TelamonStyle.reducedMotion
        NumberAnimation { duration: 90 }
    }

    Item {
        id: edges

        anchors.fill: parent
        opacity: win.active ? 0.55 + 0.45 * win.eased : 0

        Behavior on opacity {
            enabled: !TelamonStyle.reducedMotion
            NumberAnimation { id: fade; duration: 180 }
        }

        component Edge: Rectangle {
            property bool horizontal: false
            property bool flip: false

            gradient: Gradient {
                orientation: horizontal ? Gradient.Horizontal : Gradient.Vertical
                GradientStop { position: 0; color: flip ? "transparent" : win.gold }
                GradientStop { position: 0.35; color: Qt.rgba(win.bronze.r, win.bronze.g, win.bronze.b, 0.55) }
                GradientStop { position: 1; color: flip ? win.gold : "transparent" }
            }
        }

        Edge {
            anchors { left: parent.left; right: parent.right; top: parent.top }
            height: win.reach
        }
        Edge {
            anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
            height: win.reach
            flip: true
        }
        Edge {
            anchors { top: parent.top; bottom: parent.bottom; left: parent.left }
            width: win.reach
            horizontal: true
        }
        Edge {
            anchors { top: parent.top; bottom: parent.bottom; right: parent.right }
            width: win.reach
            horizontal: true
            flip: true
        }
    }

    SequentialAnimation on pulse {
        running: win.thinking && win.visible && !TelamonStyle.reducedMotion
        loops: Animation.Infinite
        onRunningChanged: if (!running) win.pulse = 0
        NumberAnimation { to: 0.6; duration: 700; easing.type: Easing.InOutSine }
        NumberAnimation { to: 0.1; duration: 700; easing.type: Easing.InOutSine }
    }
}
