// difmapy.app: a SwiftUI program around the difmapy session, which runs in the Python it embeds (pybridge.c).
//
// The window, toolbar, menus, inspector of controls, output, file panels and settings are SwiftUI and AppKit; the
// interactive plots are difmapy's own pyqtgraph plot windows, hosted in the panes (Qt runs as a guest of the app,
// see difmapy.macapp). The executable is built by packaging/macos/build_app.py against the current macOS SDK.

import AppKit
import SwiftUI

@main
struct DifmapyApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var session = Session.shared

    init() {
        // A file named on the command line is opened by the session itself (Session.start); left to AppKit it is an
        // "open document" launch, for which SwiftUI shows no window. Registered, never stored: in the app's saved
        // preferences the same setting stops the system's open panel from appearing at all (the panel service fails
        // with "listener 'com.apple.view-bridge': Connection interrupted"), so a copy saved by an earlier build goes.
        let defaults = UserDefaults.standard
        defaults.removeObject(forKey: "NSTreatUnknownArgumentsAsOpen")
        defaults.register(defaults: ["NSTreatUnknownArgumentsAsOpen": false])
    }

    var body: some Scene {
        Window("difmapy", id: "main") {
            ContentView()
                .environment(session)
                .frame(minWidth: 960, minHeight: 640)
        }
        .defaultSize(width: 1640, height: 1040)
        .commands { DifmapyCommands(session: session) }

        Settings {
            SettingsView().environment(session)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        // $DIFMAPY_APPEARANCE fixes light or dark, as for difmapy's Qt windows; otherwise the system's. Set before
        // any window exists, as the system's own would be.
        switch ProcessInfo.processInfo.environment["DIFMAPY_APPEARANCE"]?.lowercased() {
        case "dark": NSApp.appearance = NSAppearance(named: .darkAqua)
        case "light": NSApp.appearance = NSAppearance(named: .aqua)
        default: break
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Session.shared.startSoon()
    }

    /// One window, one session: closing it quits.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// Files dropped on the app's icon, or opened with it from the Finder.
    func application(_ application: NSApplication, open urls: [URL]) {
        guard let url = urls.first else { return }
        Session.shared.open(path: url.path)
        // Launched to open a file, SwiftUI has shown no window yet.
        Session.shared.showMainWindow()
    }

    /// Take the plots out of the window and close them while everything is still alive.
    func applicationWillTerminate(_ notification: Notification) {
        Session.shared.shutdown()
    }
}

struct DifmapyCommands: Commands {
    let session: Session
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        // The one handle AppKit code has on SwiftUI's windows (Session.showMainWindow).
        let _ = session.openWindow = openWindow
        CommandGroup(replacing: .newItem) {
            Button("Open…") { session.openPanel() }
                .keyboardShortcut("o")
                .disabled(!session.started || session.busy)
        }
        CommandGroup(replacing: .saveItem) {
            Menu("Save") {
                ForEach(session.savers) { saver in
                    Button("\(saver.label)…") { session.save(saver.id) }
                }
            }
            .disabled(!session.loaded || session.busy)
        }
        CommandGroup(before: .toolbar) {
            Button("Reload All Plots") { session.reloadAll() }
                .keyboardShortcut("r")
                .disabled(!session.loaded || session.busy)
            Divider()
        }
        CommandMenu("Calibrate") {
            let idle = session.loaded && !session.busy
            Button("Self-calibrate") { session.selfcal() }.disabled(!idle || !session.hasModel)
            Button(session.bayes ? "Bayesian Gain Scale" : "Gain Scale") { session.gscale() }
                .disabled(!idle || !session.hasModel)
            Divider()
            Button("Smart Flag") { session.smartFlag() }.disabled(!idle)
        }
    }
}
