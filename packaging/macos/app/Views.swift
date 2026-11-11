// The window: four panes - three of plots (the pyqtgraph plots, hosted) and the output - with the controls in an
// inspector at the side, and a toolbar.

import AppKit
import SwiftUI

struct ContentView: View {
    @Environment(Session.self) private var session
    @State private var showInspector = true

    var body: some View {
        @Bindable var session = session
        VSplitView {
            HSplitView {
                PlotPane(pane: "time").frame(minWidth: 320, minHeight: 240)
                PlotPane(pane: "uv").frame(minWidth: 320, minHeight: 240)
            }
            HSplitView {
                PlotPane(pane: "map").frame(minWidth: 320, minHeight: 300)
                OutputPane().frame(minWidth: 260, minHeight: 300)
            }
        }
        .navigationTitle("difmapy")
        .navigationSubtitle(session.filename ?? "")
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button { session.openPanel() } label: { Label("Open", systemImage: "folder") }
                    .help("Open a UVFITS file or a Measurement Set")
                    .disabled(!session.started || session.busy)
            }
            ToolbarItemGroup(placement: .primaryAction) {
                SaveMenu()
                Button { showInspector.toggle() } label: { Label("Inspector", systemImage: "sidebar.trailing") }
                    .help(showInspector ? "Hide the controls" : "Show the controls")
            }
        }
        .inspector(isPresented: $showInspector) {
            InspectorView().inspectorColumnWidth(min: 300, ideal: 350, max: 500)
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first, session.started, !session.busy else { return false }
            session.open(path: url.path)
            return true
        }
        .alert("difmapy", isPresented: Binding(get: { session.alertMessage != nil },
                                               set: { if !$0 { session.alertMessage = nil } })) {
            Button("OK") { session.alertMessage = nil }
        } message: {
            Text(session.alertMessage ?? "")
        }
        .alert("Install the difmapy command for the Terminal?", isPresented: $session.commandOffer) {
            Button("Install") { session.answerCommandOffer(install: true) }
            Button("Not Now", role: .cancel) { session.answerCommandOffer(install: false) }
        } message: {
            Text("This links \(session.commandLink) to this app, so that typing difmapy in a Terminal starts the "
                 + "interactive session (macOS may ask for your password). You can do it later in Settings.")
        }
        .task { session.startSoon() }
    }
}

struct SaveMenu: View {
    @Environment(Session.self) private var session

    var body: some View {
        Menu {
            ForEach(session.savers) { saver in
                Button("\(saver.label)…") { session.save(saver.id) }
            }
        } label: {
            Label("Save", systemImage: "square.and.arrow.down")
        }
        .help("Save a product of the session")
        .disabled(!session.loaded || session.busy)
    }
}

// MARK: the plot panes

/// The bar over each pane: what it shows in the middle (the plot picker, or a title), an action at the end.
struct PaneHeader<Center: View, Trailing: View>: View {
    @ViewBuilder var center: Center
    @ViewBuilder var trailing: Trailing

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                center
                HStack {
                    Spacer()
                    trailing
                }
                .padding(.horizontal, 8)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 34)
            .background(.bar)
            Divider()
        }
    }
}

/// One plot pane: a segmented control to pick the plot, and the plot itself - a pyqtgraph plot window, hosted. Before
/// any data are loaded it says so (the map pane offers to open a file).
struct PlotPane: View {
    let pane: String
    @Environment(Session.self) private var session

    var body: some View {
        let tabs = session.panes[pane] ?? []
        VStack(spacing: 0) {
            PaneHeader {
                if tabs.count > 1 {
                    Picker("Plot", selection: Binding(get: { session.selection[pane] ?? 0 },
                                                      set: { session.select(pane: pane, index: $0) })) {
                        ForEach(Array(tabs.enumerated()), id: \.offset) { index, name in
                            Text(name).tag(index)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                    .disabled(!session.loaded)
                } else if let name = tabs.first {
                    Text(name).font(.headline).foregroundStyle(session.loaded ? .primary : .secondary)
                }
            } trailing: {
                EmptyView()
            }
            if session.loaded {
                QtHost(pane: pane)
            } else if pane == "map" {
                ContentUnavailableView {
                    Label(session.busy ? session.busyLabel : "No Data", systemImage: "antenna.radiowaves.left.and.right")
                } description: {
                    Text(session.busy ? "" : "Open a UVFITS file or a Measurement Set, or drop one here.")
                } actions: {
                    if session.busy {
                        ProgressView().controlSize(.small)
                    } else {
                        Button("Open…") { session.openPanel() }
                            .buttonStyle(.borderedProminent)
                            .disabled(!session.started)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Text("No data")
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

/// An NSView the plots of one pane live in: the pane's Qt panel stays a top-level Qt window, and its own NSView
/// (difmapy.macapp.Session.attach hands it over) is put in here and kept at this view's size - Qt's "embedded" mode.
/// The panel is a Qt.SubWindow, so Qt never makes an NSWindow for it (from Qt 6.12, showing any other top level puts
/// its view back into one). Not a child of a foreign QWindow made from this view: Qt sends the keys typed into a
/// widget window to the top of its QWindow parents, which would be that foreign window, and the keys would go nowhere.
struct QtHost: NSViewRepresentable {
    let pane: String

    func makeNSView(context: Context) -> QtContainerView { QtContainerView(pane: pane) }
    func updateNSView(_ view: QtContainerView, context: Context) {}
    static func dismantleNSView(_ view: QtContainerView, coordinator: ()) { view.detach() }
}

final class QtContainerView: NSView {
    /// The containers in the window, so that the plots can be taken out of all of them before the app quits.
    static let live = NSHashTable<QtContainerView>.weakObjects()

    let pane: String
    private var qtView: NSView?

    init(pane: String) {
        self.pane = pane
        super.init(frame: .zero)
        Self.live.add(self)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("not used") }

    override var isFlipped: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { attach() } else { detach() }
    }

    /// Put the pane's Qt view in here, then let Qt show it.
    private func attach() {
        guard qtView == nil, let address = Session.shared.call("attach", ["pane": pane]) as? Int,
              let pointer = UnsafeMutableRawPointer(bitPattern: address) else { return }
        let view = Unmanaged<NSView>.fromOpaque(pointer).takeUnretainedValue()
        view.frame = bounds
        view.autoresizingMask = [.width, .height]
        addSubview(view)
        qtView = view
        Session.shared.call("show", ["pane": pane, "width": Int(bounds.width), "height": Int(bounds.height)])
        fitSubviews()
    }

    /// Before the view goes: hide the plots and take their view out of it.
    func detach() {
        guard let view = qtView else { return }
        qtView = nil
        Session.shared.call("detach", ["pane": pane])
        view.removeFromSuperview()
    }

    private func fitSubviews() {
        for view in subviews {
            view.autoresizingMask = [.width, .height]
            if view.frame != bounds { view.frame = bounds }
        }
    }

    override func resizeSubviews(withOldSize oldSize: NSSize) {
        super.resizeSubviews(withOldSize: oldSize)
        fitSubviews()
    }

    override func layout() {
        super.layout()
        fitSubviews()
    }
}

// MARK: the inspector

/// The side panel: the commands and their options (what they print goes to the Output pane).
struct InspectorView: View {
    var body: some View {
        VStack(spacing: 0) {
            PaneHeader {
                Text("Commands").font(.headline)
            } trailing: {
                EmptyView()
            }
            ControlsForm()
        }
    }
}

struct ControlsForm: View {
    @Environment(Session.self) private var session

    var body: some View {
        @Bindable var s = session
        let idle = s.started && !s.busy
        Form {
            Section {
                LabeledContent("File") {
                    HStack {
                        Text(s.filePath.isEmpty ? "None" : (s.filePath as NSString).lastPathComponent)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(s.filePath.isEmpty ? .secondary : .primary)
                            .help(s.filePath)
                        Button("Choose…") { s.openPanel(loadNow: false) }.disabled(!idle)
                    }
                }
                TextField("Time average", text: $s.timeAverage, prompt: Text("none (10s, 2min)"))
                TextField("Frequency average", text: $s.frequencyAverage, prompt: Text("none (channels, all)"))
                ActionRow(title: "Load", running: s.busyCommand == "load", enabled: idle && !s.filePath.isEmpty) {
                    s.load()
                }
                if s.loaded {
                    LabeledContent("Observation") { Text(s.summary).multilineTextAlignment(.trailing) }
                    LabeledContent("Model", value: s.modelSummary)
                }
            } header: {
                SectionTitle("Data")
            }

            Section {
                Toggle("Phase", isOn: $s.selfcalPhase)
                Toggle("Amplitude", isOn: $s.selfcalAmplitude)
                Toggle("Converge with modelfit", isOn: $s.converge)
                    .help("Alternate selfcal and modelfit until the model-data fit stops improving (needs fitted "
                          + "components; a CLEAN-only model gets one selfcal round)")
                TextField("Solution interval", text: $s.solint, prompt: Text("per integration (30s, scan)"))
                ActionRow(title: "Selfcal", running: s.busyCommand == "selfcal", enabled: idle && s.hasModel) {
                    s.selfcal()
                }
            } header: {
                SectionTitle("Self-calibration")
            } footer: {
                if s.loaded && !s.hasModel {
                    Text("Needs a model: CLEAN (c) or fit (m, f) one in the map.").foregroundStyle(.secondary)
                }
            }

            Section {
                Toggle("Float scale", isOn: $s.floatScale)
                    .help("Let the flux scale float onto the model's instead of keeping the data's (only for a model "
                          + "whose flux density you trust more than the a-priori calibration)")
                Toggle("Bayes", isOn: $s.bayes)
                    .help("Bayesian gscale (gscale(bayes=True)): each station left out of the source model in turn, the "
                          + "source models ticked below compared, and the corrections shrunk towards the a-priori "
                          + "calibration")
                ForEach(bayesModelNames, id: \.key) { model in
                    Toggle(isOn: Binding(get: { s.bayesModels[model.key] ?? false },
                                         set: { s.bayesModels[model.key] = $0 })) {
                        Text(model.label).padding(.leading, 14)
                    }
                    .disabled(!s.bayes)
                }
                ActionRow(title: "Gscale", running: s.busyCommand == "gscale", enabled: idle && s.hasModel) {
                    s.gscale()
                }
            } header: {
                SectionTitle("Gain scale")
            } footer: {
                Text("With Bayes, the source models the Bayesian gscale compares.").foregroundStyle(.secondary)
            }

            Section {
                Toggle("Test phases", isOn: $s.testPhases)
                    .help("Test the phases as well as the amplitudes (off: amplitudes only)")
                TextField("Antenna sigma", text: $s.antennaSigma, prompt: Text("5"))
                    .help("Threshold of an antenna's combined statistic, in Gaussian sigmas")
                    .onSubmit { if idle && s.loaded { s.smartFlag() } }
                ActionRow(title: "Smart Flag", running: s.busyCommand == "smartflag", enabled: idle && s.loaded) {
                    s.smartFlag()
                }
            } header: {
                SectionTitle("Flagging")
            } footer: {
                Text("All the data. Shift+S and a box in vplot or radplot flags a selection.")
                    .foregroundStyle(.secondary)
            }

            Section {
                TextField("UV min", text: uvField("uvmin"), prompt: Text("0"))
                    .onSubmit { if idle && s.loaded { s.applyUVRange() } }
                TextField("UV max", text: uvField("uvmax"), prompt: Text("no limit"))
                    .onSubmit { if idle && s.loaded { s.applyUVRange() } }
                ActionRow(title: "Apply UV Range", running: s.busyCommand == "uvrange", enabled: idle && s.loaded) {
                    s.applyUVRange()
                }
                TextField("Taper value", text: uvField("value"), prompt: Text("none"))
                    .help("The taper at the radius, between 0 and 1; empty for no taper")
                    .onSubmit { if idle && s.loaded { s.applyTaper() } }
                TextField("Taper radius", text: uvField("radius"), prompt: Text(""))
                    .help("The UV radius where the taper falls to the value")
                    .onSubmit { if idle && s.loaded { s.applyTaper() } }
                ActionRow(title: "Apply Taper", running: s.busyCommand == "uvtaper", enabled: idle && s.loaded) {
                    s.applyTaper()
                }
            } header: {
                SectionTitle("Imaging")
            } footer: {
                Text("Mλ, or with a unit: 500k, 2e7 lambda.").foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    /// A UV range or taper field: once typed into, the state no longer overwrites it until it is applied.
    private func uvField(_ key: String) -> Binding<String> {
        Binding(get: { session.uvFields[key] ?? "" },
                set: {
                    session.uvFields[key] = $0
                    session.uvEdited.insert(key)
                })
    }
}

/// A form row holding one action button, at the trailing edge as macOS puts them, with a spinner beside it while
/// its command runs.
struct ActionRow: View {
    let title: String
    var running = false
    var enabled = true
    let action: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Spacer()
            if running {
                ProgressView().controlSize(.small)
            }
            Button(title, action: action).disabled(!enabled)
        }
    }
}

/// The heading of a group of commands in the inspector: larger than a form's own section headers.
struct SectionTitle: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.title3.weight(.semibold))
            .foregroundStyle(.primary)
            .textCase(nil)
    }
}

// MARK: the output

/// The bottom-right pane: everything the session prints - reports, refusals, errors.
struct OutputPane: View {
    @Environment(Session.self) private var session

    var body: some View {
        VStack(spacing: 0) {
            PaneHeader {
                Text("Output").font(.headline)
            } trailing: {
                Button { session.clearOutput() } label: { Label("Clear", systemImage: "trash") }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .help("Clear the output")
            }
            ConsoleView(chunks: session.outputChunks, generation: session.outputGeneration)
        }
    }
}

/// A read-only text view that only appends what is new: the session's output can run to many thousands of lines.
struct ConsoleView: NSViewRepresentable {
    let chunks: [String]
    let generation: Int

    final class Coordinator {
        var shown = 0
        var generation = -1
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.drawsBackground = true
        scroll.borderType = .noBorder
        if let text = scroll.documentView as? NSTextView {
            text.isEditable = false
            text.isSelectable = true
            text.isRichText = false
            text.drawsBackground = true
            text.backgroundColor = .textBackgroundColor
            text.textContainerInset = NSSize(width: 6, height: 8)
            text.font = Self.font
        }
        return scroll
    }

    static let font = NSFont.monospacedSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular)

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let text = scroll.documentView as? NSTextView, let storage = text.textStorage else { return }
        let state = context.coordinator
        if state.generation != generation {
            state.generation = generation
            state.shown = 0
            storage.setAttributedString(NSAttributedString())
        }
        guard state.shown < chunks.count else { return }
        let atEnd = scroll.verticalScroller.map { $0.floatValue >= 0.999 || $0.isHidden } ?? true
        let attributes: [NSAttributedString.Key: Any] = [.font: Self.font, .foregroundColor: NSColor.textColor]
        storage.beginEditing()
        for chunk in chunks[state.shown...] {
            storage.append(NSAttributedString(string: chunk, attributes: attributes))
        }
        storage.endEditing()
        state.shown = chunks.count
        if atEnd { text.scrollToEndOfDocument(nil) }
    }
}

// MARK: settings

struct SettingsView: View {
    @Environment(Session.self) private var session

    var body: some View {
        Form {
            Section {
                LabeledContent("difmapy command", value: session.commandInstalled ? "Installed" : "Not installed")
                HStack {
                    Spacer()
                    Button(session.commandInstalled ? "Install Again…" : "Install…") { session.installCommand() }
                        .disabled(!session.commandAvailable)
                }
            } header: {
                Text("Terminal")
            } footer: {
                Text("Links \(session.commandLink) to this app, so that typing difmapy in a Terminal starts the "
                     + "interactive session with the app's own Python. macOS may ask for your password.")
                    .foregroundStyle(.secondary)
            }
            Section("About") {
                LabeledContent("Version", value: session.version.isEmpty ? "–" : session.version)
            }
        }
        .formStyle(.grouped)
        .frame(width: 460)
        .fixedSize(horizontal: false, vertical: true)
        .onAppear { session.refreshCommandStatus() }
    }
}
