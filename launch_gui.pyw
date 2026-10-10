"""由快捷方式使用项目 pythonw.exe 启动，不依赖系统 .pyw 文件关联。"""

from gui_startup import run_gui


def launch():
    # 必须晚于日志和单实例锁，才能记录依赖导入失败。
    from gui_control import main
    return main()


if __name__ == "__main__":
    raise SystemExit(run_gui(launch))
