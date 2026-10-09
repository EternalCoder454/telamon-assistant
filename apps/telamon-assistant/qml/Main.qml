pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import Telamon.Ui

// The window: whether Telamon is on, and what it last heard and said. The
// glow is a window of its own on every screen (GlowWindow).
TelamonWindow {
    id: root

    // Set from main.cpp through setInitialProperties(); see src/lib.rs.
    required property var assistant
    // Started at login: no window until the user opens Telamon.
    required property bool background

    readonly property string phase: root.assistant.phase
    readonly property bool glowing: root.phase === "awake" || root.phase === "thinking" || root.phase === "speaking"

    title: qsTr("Telamon")
    stateKey: "main"
    width: Kirigami.Units.gridUnit * 30
    height: Kirigami.Units.gridUnit * 30
    minimumWidth: Kirigami.Units.gridUnit * 22
    minimumHeight: Kirigami.Units.gridUnit * 22
    visible: !root.background

    // Closed while on: Telamon keeps listening. Off: nothing left to do.
    onClosing: {
        if (!root.assistant.enabled) {
            Qt.quit();
        }
    }

    function statusText() {
        switch (root.phase) {
        case "loading":
            return qsTr("Getting Ready…");
        case "listening":
            return qsTr("Listening for “Hey Telamon”");
        case "awake":
            return qsTr("Listening…");
        case "thinking":
            return qsTr("Thinking…");
        case "speaking":
            return qsTr("Speaking");
        case "error":
            return qsTr("Telamon Stopped");
        default:
            return qsTr("Telamon Is Off");
        }
    }

    Instantiator {
        model: Qt.application.screens
        delegate: GlowWindow {
            required property var modelData

            screen: modelData
            active: root.glowing
            level: root.assistant.level
            thinking: root.phase === "thinking"
        }
    }

    TelamonPage {
        anchors.fill: parent
        title: qsTr("Telamon")
        maxContentWidth: Kirigami.Units.gridUnit * 28

        ColumnLayout {
            Layout.fillWidth: true
            Layout.topMargin: TelamonStyle.spacingXLarge
            Layout.bottomMargin: TelamonStyle.spacingLarge
            spacing: TelamonStyle.spacing

            Glow {
                Layout.alignment: Qt.AlignHCenter
                implicitWidth: Kirigami.Units.gridUnit * 6
                implicitHeight: Kirigami.Units.gridUnit * 6
                radius: width / 2
                active: root.assistant.enabled
                level: root.assistant.level
                thinking: root.phase === "thinking"

                Symbol {
                    anchors.centerIn: parent
                    icon: Symbols.SmartToy
                    color: "#FFF8E6"
                }
            }
            TelamonLabel {
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignHCenter
                textStyle: TelamonLabel.Title
                text: root.statusText()
            }
            TelamonLabel {
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
                textStyle: TelamonLabel.Caption
                text: root.assistant.enabled
                    ? qsTr("Everything is heard and answered on this computer. Telamon's tools can only look, never change anything.")
                    : qsTr("Telamon listens for its name and answers out loud. It stays off, with the microphone closed, until you allow it.")
            }
            PrimaryButton {
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: TelamonStyle.spacing
                visible: !root.assistant.enabled
                text: qsTr("Allow Microphone and Turn On")
                onClicked: root.assistant.enable()
            }
            SecondaryButton {
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: TelamonStyle.spacing
                visible: root.assistant.enabled
                text: qsTr("Turn Off")
                onClicked: root.assistant.disable()
            }
        }

        Section {
            title: qsTr("Last Exchange")
            visible: root.assistant.heard.length > 0 || root.assistant.error.length > 0

            SectionRow {
                visible: root.assistant.heard.length > 0
                title: qsTr("You Asked")
                subtitle: root.assistant.heard
                leading: [
                    Symbol {
                        icon: Symbols.Chat
                        color: TelamonStyle.accent
                    }
                ]
            }
            SectionRow {
                visible: root.assistant.reply.length > 0
                title: qsTr("Telamon Said")
                subtitle: root.assistant.reply
                leading: [
                    Symbol {
                        icon: Symbols.AutoAwesome
                        color: TelamonStyle.accent
                    }
                ]
            }
            SectionRow {
                visible: root.assistant.error.length > 0
                title: qsTr("Problem")
                subtitle: root.assistant.error
                leading: [
                    Symbol {
                        icon: Symbols.Error
                        color: Kirigami.Theme.negativeTextColor
                    }
                ]
            }
        }
    }
}
