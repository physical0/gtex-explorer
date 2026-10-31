"""Launch the offline desktop application."""
import ctypes
import os
import sys
import traceback
from gtex_explorer.desktop_runtime import main, state_directory, TITLE


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    # Windowed executables have no console streams. Libraries still expect writable streams.
    if sys.stdout is None or sys.stderr is None:
        log_stream = (state_directory() / 'gtex_explorer.desktop_runtime.log').open("a", encoding="utf8", buffering=1)
        sys.stdout = sys.stderr = log_stream
    try:
        raise SystemExit(main())
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        (state_directory() / "last_error.txt").write_text(error, encoding="utf8")
        if os.name == "nt" and not any(x in sys.argv for x in ['--serve', '--check', '--check-training', '--training-worker']):
            ctypes.windll.user32.MessageBoxW(None,
                "Aplikasi gagal dimulai. Pastikan Microsoft Edge WebView2 Runtime tersedia.\n"
                f"Rincian kesalahan: {state_directory() / 'last_error.txt'}",
                TITLE, 0x10)
        raise SystemExit(1)
