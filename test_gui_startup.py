"""无硬件启动回归：跨进程互斥、无控制台错误日志、正常退出。"""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import uuid

import gui_startup


class StartupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows DPI / GLFW")
    def test_glfw_cycles_preserve_gui_scaling(self):
        # DPI 是不可逆的进程设置，必须用新进程验证；使用真实 GLFW 隐藏窗口。
        code = textwrap.dedent('''
            import ctypes
            from concurrent.futures import ThreadPoolExecutor
            from pathlib import Path
            import sys
            from tkinter import font
            from unittest.mock import patch

            import glfw
            import gui_control as gui
            from gui_startup import configure_window_scaling

            shcore = ctypes.WinDLL('shcore')
            shcore.GetProcessDpiAwareness.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            shcore.GetProcessDpiAwareness.restype = ctypes.c_long
            mode = int(sys.argv[2])
            if mode >= 0:
                assert shcore.SetProcessDpiAwareness(mode) == 0

            original_init = gui.tk.Tk.__init__
            def hidden_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.withdraw()

            def snapshot(app):
                app.update()
                awareness = ctypes.c_int(-1)
                assert shcore.GetProcessDpiAwareness(
                    None, ctypes.byref(awareness)) == 0
                return (awareness.value, app.tk.call('tk', 'scaling'),
                        app.winfo_screenwidth(), app.winfo_screenheight(),
                        app.winfo_reqwidth(), app.winfo_reqheight(),
                        font.nametofont('TkDefaultFont').metrics(),
                        app.status_label.winfo_reqwidth(),
                        app.status_label.winfo_reqheight())

            def open_window():
                assert glfw.init(), 'GLFW initialization failed'
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
                window = glfw.create_window(400, 300, 'DPI regression', None, None)
                assert window, 'GLFW window creation failed'
                glfw.poll_events()
                return window

            def close_window(window):
                glfw.destroy_window(window)
                glfw.terminate()

            with patch.object(gui.tk.Tk, '__init__', hidden_init), \\
                    patch.object(gui.App, '_refresh_cameras'), \\
                    patch.object(gui, '__file__', str(Path(sys.argv[1]) / 'gui_control.py')), \\
                    patch.object(gui.ROHand, 'open', side_effect=AssertionError('hardware access')):
                app = gui.App(gui.Calibration.load())
                try:
                    before = snapshot(app)
                    assert before[0] == max(0, mode), before
                    # 重复调用及已由宿主设置 DPI 的入口也应安全。
                    configure_window_scaling()
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        for cycle in range(2):
                            window = pool.submit(open_window).result(timeout=10)
                            try:
                                during = snapshot(app)
                                assert during == before, (cycle, before, during)
                            finally:
                                pool.submit(close_window, window).result(timeout=10)
                            after = snapshot(app)
                            assert after == before, (cycle, before, after)
                finally:
                    app._on_close()
                    app.mainloop()
                assert not app.worker.is_alive()
        ''')
        for mode in (-1, 1):  # 默认 Python 进程；宿主已启用 system-aware。
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                result = subprocess.run(
                    [sys.executable, "-c", code, folder, str(mode)],
                    cwd=gui_startup.ROOT, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0,
                                 result.stderr.decode(errors="replace"))

    def test_pythonw_errors_are_logged_before_dialog(self):
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="dexhand 启动 ") as folder:
            root = Path(folder)
            def failing_entry():
                self.assertEqual(Path.cwd(), root)
                print("导入前的输出")
                raise ModuleNotFoundError("测试缺少 SDK")

            def check_dialog(message, **_):
                logs = list((root / "logs").glob("startup_*.log"))
                content = logs[0].read_text(encoding="utf-8")
                self.assertIn("ModuleNotFoundError: 测试缺少 SDK", content)
                self.assertIn("导入前的输出", content)
                self.assertIn(str(logs[0]), message)

            with patch.object(gui_startup, "ROOT", root), \
                    patch.object(gui_startup, "SingleInstance"), \
                    patch.object(gui_startup, "show_message", side_effect=check_dialog) as dialog, \
                    patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                self.assertEqual(gui_startup.run_gui(failing_entry), 1)
                dialog.assert_called_once()
                self.assertIsNone(sys.stdout)
                self.assertIsNone(sys.stderr)
            self.assertEqual(Path.cwd(), previous_cwd)

    @unittest.skipUnless(os.name == "nt", "Windows named mutex")
    def test_mutex_excludes_another_process_then_releases(self):
        name = "Local\\DexHand_Test_" + uuid.uuid4().hex
        code = (
            "import sys; from gui_startup import SingleInstance, AlreadyRunningError\n"
            "try:\n"
            "    with SingleInstance(sys.argv[1]): pass\n"
            "except AlreadyRunningError: sys.exit(2)\n"
        )
        def child():
            return subprocess.run([sys.executable, "-c", code, name],
                                  cwd=gui_startup.ROOT, capture_output=True,
                                  timeout=15).returncode
        with gui_startup.SingleInstance(name):
            self.assertEqual(child(), 2)
        self.assertEqual(child(), 0)

    @unittest.skipUnless(os.name == "nt", "Windows named mutex")
    def test_mutex_released_when_owner_process_exits_abruptly(self):
        name = "Local\\DexHand_Test_" + uuid.uuid4().hex
        code = ("import os, sys; from gui_startup import SingleInstance; "
                "lock = SingleInstance(sys.argv[1]); lock.__enter__(); os._exit(7)")
        result = subprocess.run([sys.executable, "-c", code, name],
                                cwd=gui_startup.ROOT, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 7)
        with gui_startup.SingleInstance(name):
            pass

    def test_selftest_uses_normal_shutdown_without_hardware(self):
        # 让真机连接一旦被调用便失败，同时检查 selftest 关闭后的工作线程。
        code = (
            "import sys; sys.path.insert(0, sys.argv[1])\n"
            "from unittest.mock import patch\n"
            "import gui_control as gui\n"
            "created = []\n"
            "original = gui.App\n"
            "class app(original):\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        super().__init__(*args, **kwargs); created.append(self)\n"
            "sys.argv = ['gui_control.py', '--selftest']\n"
            "with patch.object(gui.ROHand, 'open', side_effect=AssertionError('hardware access')), "
            "patch.object(gui, 'App', app):\n"
            "    assert gui.main() == 0\n"
            "assert created and not created[0].worker.is_alive()\n"
        )
        with tempfile.TemporaryDirectory(prefix="dexhand cwd ") as folder:
            result = subprocess.run([sys.executable, "-c", code, str(gui_startup.ROOT)],
                                    cwd=folder, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
