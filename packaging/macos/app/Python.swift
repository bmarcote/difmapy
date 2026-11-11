// The embedded Python (pybridge.c) and the calls into difmapy.macapp.

import Foundation

/// The Python inside the app. Every call goes to `difmapy.macapp.dispatch`, with its arguments and its answer in JSON.
/// Commands that compute (`difmapy.macapp.WORKER_COMMANDS`) are run on `worker`, so the window stays live; everything
/// else, which may touch Qt, runs on the main thread.
final class Python: @unchecked Sendable {
    static let shared = Python()

    enum Reply {
        case ok(Any?)
        case failure(String)
    }

    /// Where commands that compute run.
    let worker = DispatchQueue(label: "difmapy.python.worker", qos: .userInitiated)
    private(set) var startError: String?
    private(set) var started = false

    /// Start Python: the one bundled in Contents/Resources/python, or - in a development build - the one that
    /// Contents/Resources/python-dev.json names, with its extra site directories (a venv with difmapy installed editable).
    func start() {
        guard !started else { return }
        started = true
        let bundle = Bundle.main
        // For difmapy: where the bundle is (the Terminal command links into it) and that it runs as the app.
        setenv("DIFMAPY_APP_BUNDLE", bundle.bundlePath, 1)
        unsetenv("PYTHONHOME")
        unsetenv("PYTHONPATH")
        let resources = bundle.resourceURL ?? URL(fileURLWithPath: ".")
        var home = resources.appendingPathComponent("python").path
        var executable = home + "/bin/python3"
        var sites: [String] = []
        let devFile = resources.appendingPathComponent("python-dev.json")
        if let data = try? Data(contentsOf: devFile),
           let dev = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            home = dev["home"] as? String ?? home
            executable = dev["executable"] as? String ?? executable
            sites = dev["site"] as? [String] ?? []
        }
        var error: UnsafeMutablePointer<CChar>?
        if dpy_start(home, executable, sites.joined(separator: ":"), "difmapy.macapp", pythonNotify, &error) != 0 {
            startError = error.map { String(cString: $0) } ?? "Python could not start"
            free(error)
        }
    }

    /// Run `difmapy.macapp` command `name` with keyword arguments `args`, on the calling thread.
    func call(_ name: String, _ args: [String: Any] = [:]) -> Reply {
        if let error = startError { return .failure(error) }
        guard started else { return .failure("Python is not running") }
        let data = (try? JSONSerialization.data(withJSONObject: args)) ?? Data("{}".utf8)
        guard let raw = dpy_call(name, String(decoding: data, as: UTF8.self)) else {
            return .failure("\(name): no answer from Python")
        }
        defer { dpy_free(raw) }
        let text = String(cString: raw)
        guard let reply = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] else {
            return .failure("\(name): unreadable answer \(text)")
        }
        if let error = reply["error"] as? String { return .failure(error) }
        let value = reply["ok"]
        return .ok(value is NSNull ? nil : value)
    }
}

/// What difmapy.macapp.emit sends, from whichever thread printed: handed to the session on the main thread.
let pythonNotify: dpy_notify_fn = { kind, payload in
    guard let kind, let payload else { return }
    let name = String(cString: kind)
    let json = String(cString: payload)
    DispatchQueue.main.async {
        MainActor.assumeIsolated {
            let value = (try? JSONSerialization.jsonObject(with: Data(json.utf8), options: .fragmentsAllowed)) ?? NSNull()
            Session.shared.received(name, value)
        }
    }
}
