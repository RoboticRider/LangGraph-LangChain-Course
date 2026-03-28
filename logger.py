"""Small console helpers with ANSI colors."""

from __future__ import annotations


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _out(color: str, label: str, message: str) -> None:
    print(f"{color}{Colors.BOLD}[{label}]{Colors.RESET} {message}")


def log_header(message: str) -> None:
    _out(Colors.MAGENTA, "HEADER", message)


def log_info(message: str) -> None:
    _out(Colors.CYAN, "INFO", message)


def log_success(message: str) -> None:
    _out(Colors.GREEN, "OK", message)


def log_warning(message: str) -> None:
    _out(Colors.YELLOW, "WARN", message)


def log_error(message: str) -> None:
    _out(Colors.RED, "ERROR", message)
