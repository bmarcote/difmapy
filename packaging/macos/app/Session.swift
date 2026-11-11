// The app's model: what the Python session reports, the controls' values, and the commands.

import AppKit
import Observation
import SwiftUI

struct Saver: Identifiable, Hashable {
    let id: String  // kind, a key of difmapy.plots.dashboard.SAVERS
    let label: String
    let title: String
}

/// The source models the Bayesian gscale may compare (Gscale with Bayes on), in the order shown.
let bayesModelNames: [(key: String, label: String)] = [
    ("clean", "CLEAN"), ("gauss1", "1 Gaussian"), ("gauss2", "2 Gaussians"), ("gauss3", "3 Gaussians"),
]

@MainActor @Observable
final class Session {
    static let shared = Session()

    // ---- what Python reports
    var started = false
    var version = ""
    var loaded = false
    var filename: String?
    var path: String?
    var directory: String?
    var summary = ""
    var modelSummary = ""
    var hasModel = false
    var panes: [String: [String]] = [:]
    var savers: [Saver] = []
    var saveNames: [String: String] = [:]

    // ---- the work in progress
    var starting = false
    var busy = false
    var busyLabel = ""
    /// The command running (difmapy.macapp's name for it): its button shows that it is at work.
    var busyCommand: String?
    var alertMessage: String?
    /// The output pane's text, in the pieces it was printed in; `outputGeneration` changes when it is cleared or trimmed.
    var outputChunks: [String] = []
    var outputGeneration = 0
    @ObservationIgnored private var pendingOutput = ""
    @ObservationIgnored private var outputFlushScheduled = false
    @ObservationIgnored private var outputSize = 0

    // ---- the Terminal command
    var commandOffer = false
    var commandInstalled = false
    var commandAvailable = false
    var commandLink = "/usr/local/bin/difmapy"

    // ---- the panes: which plot each shows
    var selection: [String: Int] = [:]

    // ---- the controls
    var filePath = ""
    var timeAverage = ""
    var frequencyAverage = ""
    var testPhases = true
    var antennaSigma = ""
    var selfcalPhase = true
    var selfcalAmplitude = false
    var converge = true
    var solint = ""
    var floatScale = false
    var bayes = false
    var bayesModels: [String: Bool] = ["clean": false, "gauss1": true, "gauss2": true, "gauss3": true]
    /// The UV range and taper fields (uvmin, uvmax, value, radius); those typed into keep their text until applied.
    var uvFields: [String: String] = ["uvmin": "", "uvmax": "", "value": "", "radius": ""]
    @ObservationIgnored var uvEdited: Set<String> = []

    @ObservationIgnored private var pendingOpen: String?
    @ObservationIgnored var openWindow: OpenWindowAction?
    @ObservationIgnored private var openDelegate: OpenPanelDelegate?

    var selfTest: Double? { Double(ProcessInfo.processInfo.environment["DIFMAPY_APP_SELFTEST"] ?? "") }

    // MARK: starting and stopping

    /// Start Python once the window is on screen (it takes a moment), then open what the app was started with.
    func startSoon() {
        guard !started, !starting else { return }
        starting = true
        DispatchQueue.main.async { self.start() }
    }

    func start() {
        guard !started else { return }
        Python.shared.start()
        if let error = Python.shared.startError {
            alertMessage = "Python could not start: \(error)"
            return
        }
        started = true
        apply(call("start"))
        let files = CommandLine.arguments.dropFirst().filter {
            !$0.hasPrefix("-") && FileManager.default.fileExists(atPath: $0)
        }
        if let path = pendingOpen ?? files.first {
            pendingOpen = nil
            open(path: path)
        }
        if let seconds = selfTest {
            call("selftest")
            DispatchQueue.main.asyncAfter(deadline: .now() + max(seconds - 0.3, 0.1)) {
                if let file = ProcessInfo.processInfo.environment["DIFMAPY_APP_SNAPSHOT"] { Self.snapshot(to: file) }
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { NSApp.terminate(nil) }
        } else {
            refreshCommandStatus(offer: true)
        }
    }

    /// Bring the main window up (it is not there when the app was launched to open a file).
    func showMainWindow() {
        if !NSApp.windows.contains(where: { $0.isVisible && $0.canBecomeMain }) {
            openWindow?(id: "main")
        }
        NSApp.activate()
    }

    func shutdown() {
        guard started else { return }
        for container in QtContainerView.live.allObjects { container.detach() }
        call("shutdown")
    }

    /// The main window as drawn (the plots included), for checking the look without screen-recording permission.
    static func snapshot(to file: String) {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil }),
              let view = window.contentView?.superview ?? window.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: file))
    }

    // MARK: talking to Python

    /// A command on the main thread; a failure is shown (Python has printed it to the output too).
    @discardableResult
    func call(_ name: String, _ args: [String: Any] = [:]) -> Any? {
        switch Python.shared.call(name, args) {
        case .ok(let value):
            return value
        case .failure(let message):
            report(message)
            return nil
        }
    }

    /// A command that computes, on the worker thread: the plots stop taking input meanwhile and catch up afterwards.
    func run(_ command: String, _ args: [String: Any] = [:], label: String, then: (() -> Void)? = nil) {
        guard started, !busy else { return }
        busy = true
        busyLabel = label
        busyCommand = command
        call("begin", ["label": label])
        Python.shared.worker.async {
            let reply = Python.shared.call(command, args)
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    let session = Session.shared
                    session.apply(session.call("finish", ["command": command]))
                    session.busy = false
                    session.busyCommand = nil
                    switch reply {
                    case .ok:
                        then?()
                    case .failure(let message):
                        session.report(message)
                    }
                }
            }
        }
    }

    /// A worker command that leaves the plots alone (the Terminal command).
    func background(_ command: String, _ args: [String: Any] = [:], done: @escaping (Python.Reply) -> Void) {
        Python.shared.worker.async {
            let reply = Python.shared.call(command, args)
            DispatchQueue.main.async { MainActor.assumeIsolated { done(reply) } }
        }
    }

    func report(_ message: String) {
        alertMessage = message
    }

    func received(_ kind: String, _ payload: Any) {
        switch kind {
        case "output":
            if let text = (payload as? [String: Any])?["text"] as? String { appendOutput(text) }
        case "changed":
            if !busy { refreshState() }
        default:
            break
        }
    }

    func refreshState() {
        apply(call("state"))
    }

    /// Take in the state difmapy.macapp reports.
    func apply(_ value: Any?) {
        guard let s = value as? [String: Any] else { return }
        version = s["version"] as? String ?? version
        if let p = s["panes"] as? [String: [String]] { panes = p }
        if let list = s["savers"] as? [[String]] {
            savers = list.compactMap { $0.count == 3 ? Saver(id: $0[0], label: $0[1], title: $0[2]) : nil }
        }
        loaded = s["loaded"] as? Bool ?? false
        guard loaded else { return }
        filename = s["filename"] as? String
        path = s["path"] as? String
        directory = s["directory"] as? String
        if let path { filePath = path }
        let source = s["source"] as? String ?? ""
        let nif = s["nif"] as? Int ?? 0
        let stations = s["nantennas"] as? Int ?? 0
        let pols = (s["pols"] as? [String] ?? []).joined(separator: " ")
        let flagged = (s["flagged"] as? Double ?? 0) * 100
        summary = "\(source) · \(stations) stations · \(nif) IF\(nif == 1 ? "" : "s") · \(pols) · "
            + String(format: "%.1f%% flagged", flagged)
        let ncomp = s["ncomp"] as? Int ?? 0
        hasModel = ncomp > 0
        let flux = s["model_flux"] as? Double ?? 0
        modelSummary = ncomp == 0 ? "none" : "\(ncomp) component\(ncomp == 1 ? "" : "s"), \(Self.formatFlux(flux))"
        if let uv = s["uv"] as? [String: String] {
            for (key, text) in uv where !uvEdited.contains(key) { uvFields[key] = text }
        }
        saveNames = s["save_names"] as? [String: String] ?? [:]
    }

    static func formatFlux(_ jy: Double) -> String {
        let a = abs(jy)
        if a >= 1 || a == 0 { return String(format: "%.3g Jy", jy) }
        if a >= 1e-3 { return String(format: "%.3g mJy", jy * 1e3) }
        return String(format: "%.3g µJy", jy * 1e6)
    }

    // MARK: the output

    /// Gather what is printed and show it once per pass of the run loop: a report prints in many small pieces.
    func appendOutput(_ text: String) {
        pendingOutput += text
        guard !outputFlushScheduled else { return }
        outputFlushScheduled = true
        DispatchQueue.main.async {
            self.outputFlushScheduled = false
            let text = self.pendingOutput
            self.pendingOutput = ""
            self.outputChunks.append(text)
            self.outputSize += text.utf16.count
            // Keep the last few hundred thousand characters.
            if self.outputSize > 1_000_000 {
                while self.outputSize > 600_000, !self.outputChunks.isEmpty {
                    self.outputSize -= self.outputChunks.removeFirst().utf16.count
                }
                self.outputGeneration += 1
            }
        }
    }

    func clearOutput() {
        outputChunks = []
        outputSize = 0
        outputGeneration += 1
    }

    // MARK: the panes

    func select(pane: String, index: Int) {
        selection[pane] = index
        call("select", ["pane": pane, "index": index])
    }

    func reloadAll() {
        guard loaded, !busy else { return }
        apply(call("reload"))
    }

    // MARK: files

    /// Open `path` (a UVFITS file or a Measurement Set) with the averaging set in the inspector.
    func open(path: String) {
        guard started else {
            pendingOpen = path
            return
        }
        filePath = path
        load()
    }

    func load() {
        guard !filePath.isEmpty else { return }
        let name = (filePath as NSString).lastPathComponent
        run("load", ["path": filePath, "timeavg": timeAverage, "freqavg": frequencyAverage], label: "Loading \(name)")
    }

    /// The system's open panel: a UVFITS file, or a Measurement Set - a folder, chosen as it is, or entered.
    func openPanel(loadNow: Bool = true) {
        guard started, !busy else { return }
        let panel = NSOpenPanel()
        panel.title = "Open a UVFITS file or a Measurement Set"
        panel.message = "Choose a UVFITS file, or a Measurement Set folder."
        panel.prompt = loadNow ? "Open" : "Choose"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        let start = directory ?? (filePath.isEmpty ? nil : (filePath as NSString).deletingLastPathComponent)
        if let start { panel.directoryURL = URL(fileURLWithPath: start) }
        let delegate = OpenPanelDelegate { [weak self] url in self?.chose(url, loadNow: loadNow) }
        openDelegate = delegate
        panel.delegate = delegate
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            self?.chose(url, loadNow: loadNow)
        }
    }

    private func chose(_ url: URL, loadNow: Bool) {
        filePath = url.lastPathComponent == "table.dat" ? url.deletingLastPathComponent().path : url.path
        if loadNow { load() }
    }

    /// Save one product of the session, where the system's save panel says.
    func save(_ kind: String) {
        guard loaded, !busy, let saver = savers.first(where: { $0.id == kind }) else { return }
        let panel = NSSavePanel()
        panel.title = saver.title
        panel.nameFieldStringValue = saveNames[kind] ?? ""
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        if let directory { panel.directoryURL = URL(fileURLWithPath: directory) }
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            self?.run("save", ["kind": kind, "path": url.path], label: "Saving \(saver.label.lowercased())")
        }
    }

    // MARK: flagging, calibration and imaging

    func smartFlag() {
        run("smartflag", ["use_phases": testPhases, "antenna_sigma": antennaSigma], label: "Smart flag")
    }

    func selfcal() {
        run("selfcal", ["phase": selfcalPhase, "amp": selfcalAmplitude, "converge": converge, "solint": solint],
            label: "Selfcal")
    }

    /// Gscale; with Bayes on, the Bayesian gscale over the source models ticked.
    func gscale() {
        let models = bayesModelNames.map(\.key).filter { bayesModels[$0] ?? false }
        run("gscale", ["float_scale": floatScale, "bayes": bayes, "models": models],
            label: bayes ? "Bayesian gscale" : "Gscale")
    }

    func applyUVRange() {
        run("uvrange", ["uvmin": uvFields["uvmin"] ?? "", "uvmax": uvFields["uvmax"] ?? ""], label: "UV range") {
            self.uvEdited.subtract(["uvmin", "uvmax"])
            self.refreshState()
        }
    }

    func applyTaper() {
        run("uvtaper", ["value": uvFields["value"] ?? "", "radius": uvFields["radius"] ?? ""], label: "Taper") {
            self.uvEdited.subtract(["value", "radius"])
            self.refreshState()
        }
    }

    // MARK: the Terminal command

    func refreshCommandStatus(offer: Bool = false) {
        guard started, let s = call("command_status") as? [String: Any] else { return }
        commandInstalled = s["installed"] as? Bool ?? false
        commandAvailable = s["target"] is String
        commandLink = s["link"] as? String ?? commandLink
        if offer { commandOffer = s["offer"] as? Bool ?? false }
    }

    func answerCommandOffer(install: Bool) {
        commandOffer = false
        background("answer_command_offer", ["install": install]) { reply in
            if install, case .ok(let value) = reply, let r = value as? [String: Any], r["installed"] as? Bool != true {
                self.report("The command was not installed: \(r["message"] as? String ?? "")")
            }
            self.refreshCommandStatus()
        }
    }

    func installCommand() {
        background("install_command") { reply in
            if case .failure(let message) = reply { self.report(message) }
            self.refreshCommandStatus()
        }
    }
}

/// The open panel's filter: folders (to go into, or a Measurement Set to choose) and visibility files. Going into a
/// Measurement Set chooses it, as it is a folder with nothing in it to pick.
final class OpenPanelDelegate: NSObject, NSOpenSavePanelDelegate {
    static let extensions: Set<String> = ["uvf", "uvfits", "fits"]
    let onMeasurementSet: (URL) -> Void

    init(onMeasurementSet: @escaping (URL) -> Void) {
        self.onMeasurementSet = onMeasurementSet
    }

    static func isMeasurementSet(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appendingPathComponent("table.dat").path)
    }

    func panel(_ sender: Any, shouldEnable url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return isDirectory.boolValue || Self.extensions.contains(url.pathExtension.lowercased())
            || url.lastPathComponent == "table.dat"
    }

    func panel(_ sender: Any, didChangeToDirectoryURL url: URL?) {
        guard let url, Self.isMeasurementSet(url), let panel = sender as? NSOpenPanel else { return }
        DispatchQueue.main.async {
            panel.cancel(nil)
            self.onMeasurementSet(url)
        }
    }
}
