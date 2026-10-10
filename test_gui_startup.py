"""无硬件启动回归：跨进程互斥、无控制台错误日志、正常退出。"""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import gui_startup


class StartupTests(unittest.TestCase):
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
