"""GUI 启动支持；只用标准库，在导入 SDK / 仿真依赖之前运行。"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import traceback


ROOT = Path(__file__).resolve().parent
APP_NAME = "灵巧手控制台"
# 同一 Windows 会话中共用一个控制台，包括命令行和 pythonw 启动。
MUTEX_NAME = r"Local\ROH_LiteS001_Control"


class AlreadyRunningError(Exception):
    pass


class SingleInstance:
    """持有内核句柄直到退出；异常退出也由 Windows 回收，不留下锁文件。"""

    def __init__(self, name=MUTEX_NAME):
        self.name = name
        self.handle = None

    def __enter__(self):
        if os.name != "nt":
            return self
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL,
                                             wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        handle = self.kernel.CreateMutexW(None, False, self.name)
        error = ctypes.get_last_error()
        if not handle:
            raise ctypes.WinError(error)
        if error == 183:  # ERROR_ALREADY_EXISTS
            self.kernel.CloseHandle(handle)
            raise AlreadyRunningError("灵巧手控制台已经在运行，请切换到已有窗口。")
        self.handle = handle
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def show_message(message, *, error=True):
    """不依赖 Tk，连 tkinter / SDK 导入失败时也能报告错误。"""
    if os.name == "nt":
        user = ctypes.WinDLL("user32", use_last_error=True)
        user.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR,
                                     wintypes.LPCWSTR, wintypes.UINT]
        user.MessageBoxW.restype = ctypes.c_int
        user.MessageBoxW(None, message, APP_NAME, 0x10 if error else 0x40)
    elif sys.stderr is not None:
        print(message, file=sys.stderr)


def _open_log():
    name = f"startup_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"
    folders = [ROOT / "logs"]
    if os.environ.get("LOCALAPPDATA"):
        folders.append(Path(os.environ["LOCALAPPDATA"]) / "DexHand" / "logs")
    last_error = None
    for folder in folders:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
            return path, path.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            last_error = exc
    raise OSError(f"无法创建启动日志：{last_error}")


class _Tee:
    """调试时同时写终端和文件；pythonw 没有终端时直接使用日志流。"""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.log.write(text)
        try:
            self.console.write(text)
        except (OSError, UnicodeError):
            pass
        return len(text)

    def flush(self):
        self.log.flush()
        try:
            self.console.flush()
        except OSError:
            pass

    def __getattr__(self, name):
        return getattr(self.console, name)


def run_gui(entry):
    """设置日志、占用实例锁，再调用延迟导入的 GUI 入口。"""
    try:
        log_path, log = _open_log()
    except OSError as exc:
        show_message(str(exc))
        return 1

    original_out, original_err = sys.stdout, sys.stderr
    original_thread_hook = threading.excepthook
    original_cwd = Path.cwd()
    for stream in (original_out, original_err):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stdout = _Tee(original_out, log) if original_out is not None else log
    sys.stderr = _Tee(original_err, log) if original_err is not None else log

    def thread_error(args):
        print(f"后台线程异常：{args.thread.name}", file=sys.stderr)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback,
                                  file=sys.stderr)

    threading.excepthook = thread_error
    try:
        print(f"启动时间：{datetime.now().astimezone().isoformat()}")
        print(f"Python：{sys.executable}\n项目目录：{ROOT}\n启动日志：{log_path}")
        with SingleInstance():
            os.chdir(ROOT)
            result = entry()
            print(f"GUI 已退出，退出码：{result or 0}")
            return result or 0
    except AlreadyRunningError as exc:
        print(str(exc))
        show_message(str(exc), error=False)
        return 2
    except Exception as exc:
        traceback.print_exc()
        show_message(f"启动失败：{type(exc).__name__}: {exc}\n\n"
                     f"详细信息已保存到：\n{log_path}\n\n"
                     "也可双击 debug_gui.cmd 查看调试输出。")
        return 1
    finally:
        threading.excepthook = original_thread_hook
        sys.stdout, sys.stderr = original_out, original_err
        log.close()
        os.chdir(original_cwd)
