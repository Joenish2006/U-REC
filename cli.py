"""U-REC command line. Run with no arguments for the menu.

Owner: Joenish

    python cli.py                      interactive menu
    python cli.py identify [--web]     live recognition (window, or browser at localhost)
    python cli.py register NAME        Face ID-style scan to add a person
    python cli.py list                 registered people
    python cli.py delete NAME          remove one person
    python cli.py reset                remove everyone (asks you to type YES)
    python cli.py info                 settings, devices, model folder
    python cli.py download-models      fetch models now (e.g. before an offline demo)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from urec import config, models, term
from urec.database import Database, clean_name


def _db(cfg: dict) -> Database:
    return Database(config.database_path(cfg))


def _confirm(question: str) -> bool:
    return term.ask(f"{question} {term.dim('[y/N]')} ").strip().lower() in ("y", "yes")


def _device(cfg: dict) -> str:
    from urec.runtime import providers

    first = providers(cfg["recognition"]["device"])[0]
    return {"CUDAExecutionProvider": "gpu · cuda", "DmlExecutionProvider": "gpu · directml",
            "CoreMLExecutionProvider": "gpu · coreml"}.get(first, "cpu")


# ---------------------------------------------------------------- commands

def cmd_identify(cfg: dict, camera: int | None, web: bool) -> int:
    if web:
        from urec import web as web_mod
        web_mod.serve(cfg, camera)
    else:
        from urec import identify
        term.info("Opening camera. Press q or Esc in the window to stop.")
        identify.run(cfg, camera)
    return 0


def cmd_register(cfg: dict, name: str, camera: int | None, force: bool) -> int:
    from urec import register
    from urec.embedder import get_embedder
    from urec.matcher import Matcher

    try:
        name = clean_name(name)
    except ValueError as exc:
        term.error(str(exc))
        return 1
    with _db(cfg) as db:
        if db.has(name) and not force and not _confirm(f"{name} is already registered. Scan again and replace?"):
            return 1
        term.info("Loading face model ...")
        embedder = get_embedder(cfg)
        term.info(f"Scanning {term.BOLD}{name}{term.RESET}. Follow the prompts in the camera window.")
        vectors = register.scan(cfg, embedder, name, camera)
        if vectors is None:
            term.error("Nothing saved.")
            return 1

        others = {n: v for n, v in db.gallery(embedder.name).items() if n.lower() != name.lower()}
        if others:
            mean = vectors.mean(axis=0)
            match, score = Matcher(others, config.threshold(cfg)).match(mean / np.linalg.norm(mean))
            if match:
                term.warn(f"This face looks like {term.BOLD}{match}{term.RESET} (similarity {score:.2f}).")
                if not force and not _confirm(f"Save as {name} anyway?"):
                    return 1

        db.save_person(name, embedder.name, vectors)
    term.ok(f"Saved {term.BOLD}{name}{term.RESET} {term.dim(f'· {len(vectors)} face samples · {embedder.name}')}")
    return 0


def cmd_list(cfg: dict) -> int:
    backend = cfg["recognition"]["backend"]
    with _db(cfg) as db:
        people = db.people()
    if not people:
        term.info("Nobody is registered yet. Choose Register to add someone.")
        return 0
    print()
    print(f"  {term.GREY}{'NAME':<22}{'SAMPLES':<10}REGISTERED{term.RESET}")
    for name, created, counts in people:
        n = counts.get(backend, 0)
        mark = term.dot(term.GREEN if n else term.YELLOW)
        note = "" if n else term.dim(f"  re-register for {backend}")
        print(f"  {mark} {term.BOLD}{name:<20}{term.RESET}{n:<10}{term.dim(created[:10])}{note}")
    print()
    return 0


def cmd_delete(cfg: dict, name: str, yes: bool) -> int:
    with _db(cfg) as db:
        if not db.has(name):
            term.error(f"No one called {name!r} is registered.")
            return 1
        if not yes and not _confirm(f"Delete {name}?"):
            return 1
        db.delete(name)
    term.ok(f"Deleted {name}.")
    return 0


def cmd_reset(cfg: dict, yes: bool) -> int:
    if not yes:
        term.warn(f"This deletes {term.BOLD}every{term.RESET} registered person.")
        if term.ask(f"Type {term.RED}{term.BOLD}YES{term.RESET} to confirm: ").strip() != "YES":
            term.info("Reset cancelled.")
            return 1
    with _db(cfg) as db:
        n = db.reset()
    term.ok(f"Database reset · removed {n} {'person' if n == 1 else 'people'}.")
    return 0


def cmd_info(cfg: dict) -> int:
    rec = cfg["recognition"]
    with _db(cfg) as db:
        count = len(db.people())
    backend = rec["backend"] + (f" ({rec['insightface_model']})" if rec["backend"] == "insightface" else "")
    rows = [
        ("config", config.CONFIG_FILE if config.CONFIG_FILE.exists() else "defaults (no config.toml)"),
        ("backend", backend),
        ("threshold", config.threshold(cfg)),
        ("body model", cfg["bodies"]["model"]),
        ("device", _device(cfg)),
        ("models", models.model_dir()),
        ("database", f"{config.database_path(cfg)} ({count} people)"),
    ]
    print()
    for key, value in rows:
        print(f"  {term.GREY}{key:<12}{term.RESET}{value}")
    print()
    return 0


def cmd_download(cfg: dict) -> int:
    insightface = cfg["recognition"]["backend"] == "insightface"
    keys = [cfg["bodies"]["model"]] + ([] if insightface else ["yunet", "sface"])
    for key in keys:
        term.ok(f"{key:<12}{term.dim(str(models.get(key)))}")
    if insightface:
        from urec.embedder import get_embedder
        get_embedder(cfg)  # insightface downloads its own models on first load
        term.ok("insightface ready")
    return 0


# ---------------------------------------------------------------- menu

MENU = [
    ("1", "Identify", "live camera window"),
    ("2", "Identify web", "stream to localhost"),
    ("3", "Register", "add a person with a face scan"),
    ("4", "People", "list who is registered"),
    ("5", "Delete", "remove one person"),
    ("6", "Reset", "remove everyone"),
    ("q", "Quit", ""),
]


def _status(cfg: dict) -> str:
    with _db(cfg) as db:
        people = len(db.people())
    g, b = term.GREEN, term.BLUE
    return (f"  {term.dot(g)} {cfg['recognition']['backend']}   "
            f"{term.dot(g if people else term.YELLOW)} {people} registered   "
            f"{term.dot(b)} {_device(cfg)}")


def _print_menu(cfg: dict) -> None:
    print()
    print(_status(cfg))
    print(term.rule(46))
    for key, label, desc in MENU:
        print(f"   {term.GREEN}{term.BOLD}{key}{term.RESET}   {term.WHITE}{term.BOLD}{label:<14}{term.RESET}"
              f"{term.dim(desc)}")
    print()


def menu(cfg: dict, camera: int | None) -> int:
    print(term.title())
    while True:
        _print_menu(cfg)
        try:
            choice = term.ask("").strip().lower()
        except EOFError:  # Ctrl+D / closed input
            print()
            return 0
        try:
            if choice == "1":
                cmd_identify(cfg, camera, web=False)
            elif choice == "2":
                cmd_identify(cfg, camera, web=True)
            elif choice == "3":
                cmd_register(cfg, term.ask("Name: "), camera, force=False)
            elif choice == "4":
                cmd_list(cfg)
            elif choice == "5":
                cmd_list(cfg)
                cmd_delete(cfg, term.ask("Name to delete: ").strip(), yes=False)
            elif choice == "6":
                cmd_reset(cfg, yes=False)
            elif choice in ("q", "quit", "exit"):
                print(term.dim("  bye.\n"))
                return 0
            else:
                term.warn("Pick 1-6 or q.")
        except KeyboardInterrupt:
            print()
            term.info("Stopped.")
        except EOFError:
            print()
            return 0
        except Exception as exc:
            term.error(str(exc))


# ---------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="urec", description="U-REC face recognition")
    parser.add_argument("--config", help="settings file (default: config.toml)")
    parser.add_argument("--camera", type=int, help="camera index (default from config)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("identify", help="live recognition")
    p.add_argument("--web", action="store_true", help="stream to http://localhost:5000 instead of a window")
    p = sub.add_parser("register", help="scan and add a person")
    p.add_argument("name")
    p.add_argument("--force", action="store_true", help="replace without asking")
    sub.add_parser("list", help="show registered people")
    p = sub.add_parser("delete", help="remove one person")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("reset", help="remove everyone")
    p.add_argument("--yes", action="store_true", help="skip the YES prompt")
    sub.add_parser("info", help="show settings and devices")
    sub.add_parser("download-models", help="fetch models now")
    sub.add_parser("init", help="create config.toml from config.example.toml")
    p = sub.add_parser("set-backend", help="switch recognition backend in config.toml")
    p.add_argument("backend", choices=config.BACKENDS)

    args = parser.parse_args(argv)
    if args.command == "init":
        term.ok("Created config.toml" if config.init_file() else "config.toml already exists")
        return 0
    if args.command == "set-backend":
        config.set_backend(args.backend)
        term.ok(f"Backend set to {args.backend} in config.toml")
        return 0

    try:
        cfg = config.load(args.config)
        if args.command == "identify":
            return cmd_identify(cfg, args.camera, args.web)
        if args.command == "register":
            return cmd_register(cfg, args.name, args.camera, args.force)
        if args.command == "list":
            return cmd_list(cfg)
        if args.command == "delete":
            return cmd_delete(cfg, args.name, args.yes)
        if args.command == "reset":
            return cmd_reset(cfg, args.yes)
        if args.command == "info":
            return cmd_info(cfg)
        if args.command == "download-models":
            return cmd_download(cfg)
        return menu(cfg, args.camera)
    except KeyboardInterrupt:
        print()
        term.info("Stopped.")
        return 130
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        term.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
