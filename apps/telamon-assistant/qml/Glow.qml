import QtQuick
import Telamon.Ui

// Telamon's gold-to-bronze glow, inside a rounded shape (the window's orb).
// It breathes with `level` (the voice), pulses slowly while `thinking`, and
// stays still under reduced motion.
Rectangle {
    id: glow

    property bool active: false
    property real level: 0
    property bool thinking: false
    property real pulse: 0

    gradient: Gradient {
        GradientStop { position: 0; color: "#F7D774" }
        GradientStop { position: 0.6; color: "#D49A3A" }
        GradientStop { position: 1; color: "#9C5F24" }
    }
    opacity: glow.active ? 0.7 + 0.3 * Math.max(glow.level, glow.pulse) : 0.35
    scale: 1 + 0.08 * Math.max(glow.level, glow.pulse)

    Behavior on scale {
        enabled: !TelamonStyle.reducedMotion
        NumberAnimation { duration: 90 }
    }

    SequentialAnimation on pulse {
        running: glow.thinking && !TelamonStyle.reducedMotion
        loops: Animation.Infinite
        onRunningChanged: if (!running) glow.pulse = 0
        NumberAnimation { to: 0.6; duration: 700; easing.type: Easing.InOutSine }
        NumberAnimation { to: 0.1; duration: 700; easing.type: Easing.InOutSine }
    }
}
