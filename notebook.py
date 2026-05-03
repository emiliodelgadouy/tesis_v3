import importlib
import subprocess
import sys
import os
import os

from pathlib import Path


def is_running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False


def install_requirements_silent(requirements_path: str = "requirements.txt") -> None:
    req_path = Path(requirements_path).resolve()

    if not req_path.exists():
        raise FileNotFoundError(f"No existe {req_path}")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        "-r",
        str(req_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Falló la instalación de requirements.\n\n"
            f"Comando:\n{' '.join(cmd)}\n\n"
            f"Error de pip:\n{result.stderr}"
        )


def configure_notebook(
    autoreload: bool = True,
    disable_bytecode: bool = True,
    requirements_path: str = "requirements.txt",
    install_requirements: bool = True,
    skip_requirements_in_colab: bool = True,
) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if disable_bytecode:
        sys.dont_write_bytecode = True

    running_in_colab = is_running_in_colab()

    if install_requirements and not (running_in_colab and skip_requirements_in_colab):
        install_requirements_silent(requirements_path)

    importlib.invalidate_caches()

    if not autoreload:
        return

    try:
        from IPython import get_ipython
    except ImportError:
        return

    shell = get_ipython()
    if shell is None:
        return

    shell.run_line_magic("load_ext", "autoreload")
    shell.run_line_magic("autoreload", "2")