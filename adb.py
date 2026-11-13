"""ADB-based background interaction with an Android emulator (LDPlayer, MuMuPlayer, MSI/BlueStacks).

Coordinates are guest pixels (the resolution reported by ``wm size``), which
makes them independent of the emulator window's position/size on the desktop.
"""
import glob
import os
import shutil
import socket
import subprocess
import time

import cv2
import numpy as np


def _platform_tools_adb():
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        base = os.environ.get(var)
        if base:
            candidate = os.path.join(base, "platform-tools", "adb.exe")
            if os.path.isfile(candidate):
                return candidate
    return None


def _find_ldplayer_adb():
    """Look for LDPlayer's bundled adb.exe under common install locations."""
    search_roots = []
    for drive in "CDEF":
        drive += ":"
        if os.path.exists(drive):
            search_roots.append(drive + os.sep)

    for root in search_roots:
        hits = glob.glob(os.path.join(root, "LDPlayer*", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "LDPlayer9", "adb.exe"))
        if hits:
            return hits[0]
    return None


def _find_msi_adb():
    """Look for MSI App Player's bundled adb.exe under common install locations."""
    search_roots = []
    for drive in "CDEF":
        drive += ":"
        if os.path.exists(drive):
            search_roots.append(drive + os.sep)

    for root in search_roots:
        # Common paths
        hits = glob.glob(os.path.join(root, "MSI App Player", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "MSI App Player", "adb.exe"))
        hits += glob.glob(os.path.join(root, "MSI", "MSI App Player", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "Program Files", "MSI", "MSI App Player", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "Program Files (x86)", "MSI", "MSI App Player", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "*", "MSI App Player", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "BlueStacks_msi5", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "BlueStacks_msi5", "*", "HD-Adb.exe"))
        hits += glob.glob(os.path.join(root, "Program Files", "BlueStacks_msi5", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "Program Files", "BlueStacks_msi5", "*", "HD-Adb.exe"))
        hits += glob.glob(os.path.join(root, "Program Files (x86)", "BlueStacks X_msi5", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "Users", "*", "AppData", "Local", "MSI App Player", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "Users", "*", "AppData", "Local", "HD-Player_msi5", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "Users", "*", "AppData", "Local", "HD-Player_msi5", "*", "HD-Adb.exe"))
        # BlueStacks MSI variants (common for MSI Player installs)
        hits += glob.glob(os.path.join(root, "Program Files", "BlueStacks_msi5", "HD-Adb.exe"))
        hits += glob.glob(os.path.join(root, "Program Files", "BlueStacks_msi5", "*", "HD-Adb.exe"))
        if hits:
            return hits[0]
    return None


def _find_mumu_adb():
    """Look for MuMuPlayer's bundled adb.exe under common install locations."""
    search_roots = []
    for drive in "CDEF":
        drive += ":"
        if os.path.exists(drive):
            search_roots.append(drive + os.sep)

    patterns = [
        os.path.join("Program Files", "Netease", "MuMuPlayer*", "nx_main", "adb.exe"),
        os.path.join("Program Files", "Netease", "MuMuPlayer*", "nx_device", "*", "shell", "adb.exe"),
        os.path.join("Program Files", "Netease", "MuMu*", "*", "adb.exe"),
        os.path.join("Program Files (x86)", "Netease", "MuMu*", "*", "adb.exe"),
        os.path.join("Netease", "MuMu*", "*", "adb.exe"),
        os.path.join("Program Files", "MuMu*", "*", "adb.exe"),
        os.path.join("MuMu*", "*", "adb.exe"),
        # Older MuMu Player (6/9)
        os.path.join("Program Files", "Netease", "MuMu*", "emulator", "nemu", "vmonitor", "bin", "adb_server.exe"),
        os.path.join("Program Files (x86)", "Netease", "MuMu*", "emulator", "nemu", "vmonitor", "bin", "adb_server.exe"),
    ]
    for root in search_roots:
        for pattern in patterns:
            hits = glob.glob(os.path.join(root, pattern))
            if hits:
                return hits[0]
    return None


def find_adb():
    """Return a usable adb.exe path or None."""
    path = shutil.which("adb")
    if path:
        return path
    path = shutil.which("adb.exe")
    if path:
        return path
    path = _platform_tools_adb()
    if path:
        return path
    path = _find_mumu_adb()
    if path:
        return path
    path = _find_msi_adb()
    if path:
        return path
    return _find_ldplayer_adb()


class EmulatorAdb:
    """
    Thin subprocess wrapper around ``adb`` talking to one emulator instance.

    Defaults to the first LDPlayer instance (127.0.0.1:5555). If ``auto_scan``
    is enabled and the configured port is unreachable, the odd ports 5555-5599
    are probed (each additional LDPlayer instance owns the next odd port).
    """

    def __init__(self, host="127.0.0.1", port=5555, adb_path="", auto_scan=True):
        self.host = host
        self.port = int(port)
        self.adb = adb_path or find_adb()
        self.serial = f"{host}:{self.port}"
        self.auto_scan = auto_scan
        self.last_error = ""
        self._connected = False
        self._res = (0, 0)

    # ---- state ----

    @property
    def connected(self):
        return self._connected

    def connect(self, timeout=15.0):
        """Connect to the emulator and query its resolution. Returns bool."""
        if self.adb is None:
            self.last_error = (
                "adb not found (install platform-tools or use MuMuPlayer / LDPlayer / MSI adb)"
            )
            return False

        mumu_ports = [16384, 16416, 16448, 16480, 16512, 7555]
        ld_ports = list(range(5555, 5600, 2))
        is_mumu_adb = "mumu" in self.adb.lower()

        if self.auto_scan:
            ordered = (mumu_ports + ld_ports) if is_mumu_adb else ([self.port] + mumu_ports + ld_ports)
            candidates = list(dict.fromkeys(ordered))
        else:
            candidates = [self.port]

        # Fast-path: if any candidate port is already listening, try those first
        # so we don't stall for `timeout` seconds on a closed default port.
        open_ports = [p for p in candidates if self._port_open(self.host, p)]
        if open_ports:
            candidates = open_ports + [p for p in candidates if p not in open_ports]

        for index, port in enumerate(candidates):
            # A refused `adb connect` costs several seconds of subprocess
            # churn per port; probe the TCP port first so a dead emulator
            # fails fast. Only the first candidate gets the full boot wait.
            if index > 0 and not self._port_open(self.host, port):
                continue
            serial = f"{self.host}:{port}"
            self._run(["connect", serial], timeout=8)
            # The primary candidate gets full patience (emulator may still be
            # booting adbd); other auto-scanned ports get one quick check.
            deadline = time.time() + (timeout if index == 0 else 1.0)
            while time.time() < deadline:
                if self._is_device(serial):
                    self.serial = serial
                    self.port = port
                    self._connected = True
                    self._refresh_resolution()
                    return True
                time.sleep(0.2)

        self.last_error = (
            f"could not reach any adb device on {self.host} "
            "(checked ports 5555-5599, 7555, 16384-16512)"
        )
        return False

    def _is_device(self, serial):
        p = self._run(["devices"], timeout=8)
        if p.returncode != 0:
            return False
        for line in p.stdout.decode("utf-8", "ignore").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == serial and parts[1].strip() == "device":
                return True
        return False

    @staticmethod
    def _port_open(host, port, timeout=0.25):
        """Cheap TCP probe; closed ports fail fast on loopback."""
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError:
            return False

    # ---- commands ----

    def _run(self, args, timeout=30.0):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            return subprocess.run(
                [self.adb, "-s", self.serial] + args,
                capture_output=True,
                timeout=timeout,
                creationflags=flags,
            )
        except subprocess.TimeoutExpired:
            self.last_error = f"adb command timed out: {' '.join(args)}"
            return None

    def _refresh_resolution(self):
        p = self._run(["shell", "wm", "size"], timeout=10)
        if p is None or p.returncode != 0:
            return
        for word in p.stdout.decode("utf-8", "ignore").replace(":", " ").split():
            if "x" in word and word.replace("x", "").isdigit():
                try:
                    w, h = (int(v) for v in word.split("x"))
                    self._res = (w, h)
                except ValueError:
                    pass
                return

    def resolution(self):
        """Return guest resolution as (width, height)."""
        if self._res == (0, 0):
            self._refresh_resolution()
        return self._res

    def screencap(self):
        """Capture the guest framebuffer as a BGR numpy image (crash-safe)."""
        if not self._connected:
            return None
        p = self._run(["exec-out", "screencap", "-p"], timeout=30)
        if p is None or p.returncode != 0 or not p.stdout:
            self.last_error = "screencap failed"
            return None
        frame = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.last_error = "screencap decode failed"
            return None
        return frame

    def tap(self, x, y):
        """Tap at guest pixel coordinates."""
        if not self._connected:
            return False
        p = self._run(
            ["shell", "input", "tap", str(int(x)), str(int(y))], timeout=10
        )
        return p is not None and p.returncode == 0

    def keyevent(self, keycode):
        """Send a raw Android key event (e.g. 3 = HOME, 4 = BACK)."""
        if not self._connected:
            return False
        p = self._run(["shell", "input", "keyevent", str(int(keycode))], timeout=10)
        return p is not None and p.returncode == 0