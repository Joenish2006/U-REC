"""Terminal styling: the U-REC title art, colours and status messages.

Owner: Joenish

Colours turn off automatically when output isn't a terminal or NO_COLOR is set
(https://no-color.org). Old consoles that can't print the block characters get a
plain-ASCII title instead.
"""
from __future__ import annotations

import os
import sys

from . import __version__

# ---------------------------------------------------------------- colour support

def _enable_windows_ansi() -> None:
    if sys.platform == "win32":
        os.system("")  # switches the Windows console into ANSI (VT) mode


def _use_colour() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _can_print(text: str) -> bool:
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_enable_windows_ansi()
COLOUR = _use_colour()


def rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m" if COLOUR else ""


def _sgr(code: str) -> str:
    return f"\033[{code}m" if COLOUR else ""


RESET = _sgr("0")
BOLD = _sgr("1")
DIM = _sgr("2")
GREEN = rgb(63, 185, 80)
RED = rgb(248, 81, 73)
YELLOW = rgb(210, 153, 34)
BLUE = rgb(88, 166, 255)
WHITE = rgb(230, 237, 243)
GREY = rgb(139, 148, 158)

# ---------------------------------------------------------------- title art

_GLYPHS = {
    "U": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "-": ["      ", "      ", "█████╗", "╚════╝", "      ", "      "],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "C": [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
}

_PLAIN = [
    r" _   _       ____  _____ ____ ",
    r"| | | |     |  _ \| ____/ ___|",
    r"| | | |_____| |_) |  _|| |    ",
    r"| |_| |_____|  _ <| |__| |___ ",
    r" \___/      |_| \_\_____\____|",
]

# top-to-bottom gradient: light grey-white into green
_GRADIENT = [(230, 237, 243), (196, 226, 205), (160, 215, 168),
             (124, 204, 131), (93, 194, 104), (63, 185, 80)]


def _art_lines() -> list[str]:
    return ["".join(_GLYPHS[ch][row] for ch in "U-REC") for row in range(6)]


def title(subtitle: str = "real-time face recognition") -> str:
    """The launch banner: block letters framed by green detection-box corners."""
    art = _art_lines()
    fancy = _can_print("█╗┌─")
    if not fancy:
        art = _PLAIN
    width = max(len(line) for line in art)
    pad = "   "
    out: list[str] = [""]
    tag = f" v{__version__} "
    if fancy:
        out.append(f" {GREEN}┌─{RESET}{BOLD}{rgb(13, 17, 23)}{_bg(63, 185, 80)}{tag}{RESET}"
                   f"{' ' * (width + 2 * len(pad) - 4 - len(tag))}{GREEN}─┐{RESET}")
    else:
        out.append("")
    for i, line in enumerate(art):
        colour = rgb(*_GRADIENT[min(i, len(_GRADIENT) - 1)])
        out.append(f" {pad}{BOLD}{colour}{line}{RESET}")
    if fancy:
        out.append(f" {GREEN}└─{RESET}{' ' * (width + 2 * len(pad) - 4)}{GREEN}─┘{RESET}")
    out.append(f" {pad}{GREY}{subtitle}{RESET}")
    return "\n".join(out)


def _bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m" if COLOUR else ""

# ---------------------------------------------------------------- messages

def _mark(symbol: str, fallback: str) -> str:
    return symbol if _can_print(symbol) else fallback


def ok(msg: str) -> None:
    print(f"  {GREEN}{_mark('✔', '+')}{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}{_mark('!', '!')}{RESET} {msg}")


def error(msg: str) -> None:
    print(f"  {RED}{_mark('✖', 'x')}{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {BLUE}{_mark('›', '>')}{RESET} {msg}")


def dim(msg: str) -> str:
    return f"{GREY}{msg}{RESET}"


def ask(prompt: str) -> str:
    return input(f"  {GREEN}{_mark('›', '>')}{RESET} {prompt}")


def rule(width: int = 50) -> str:
    return f"  {GREY}{_mark('─', '-') * width}{RESET}"


def dot(colour: str) -> str:
    return f"{colour}{_mark('●', '*')}{RESET}"
