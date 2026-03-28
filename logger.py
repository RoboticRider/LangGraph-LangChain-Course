# --------Documentation Assist Agent------
from __future__ import annotations

import sys
from typing import Any

HEADER = "--------Documentation Assist Agent------"


class Colors:
    """ANSI terminal colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def log_header(message: str | None = None) -> None:
    """Print the agent banner, optionally with an extra line."""
    print(f"{Colors.BOLD}{Colors.CYAN}{HEADER}{Colors.RESET}", flush=True)
    if message:
        print(f"{Colors.CYAN}{message}{Colors.RESET}", flush=True)


def log_info(*args: Any, sep: str = " ", end: str = "\n") -> None:
    text = sep.join(str(a) for a in args)
    print(f"{Colors.BLUE}{text}{Colors.RESET}", end=end, flush=True)


def log_success(*args: Any, sep: str = " ", end: str = "\n") -> None:
    text = sep.join(str(a) for a in args)
    print(f"{Colors.GREEN}{text}{Colors.RESET}", end=end, flush=True)


def log_warning(*args: Any, sep: str = " ", end: str = "\n") -> None:
    text = sep.join(str(a) for a in args)
    print(f"{Colors.YELLOW}{text}{Colors.RESET}", end=end, flush=True)


def log_error(*args: Any, sep: str = " ", end: str = "\n") -> None:
    text = sep.join(str(a) for a in args)
    print(f"{Colors.RED}{text}{Colors.RESET}", end=end, file=sys.stderr, flush=True)
