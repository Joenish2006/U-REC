"""U-REC command line. Run with no arguments for the menu.

Owner: Omkar

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

from urec import config, models
from urec.database import Database, clean_name


def _db(cfg: dict) -> Database:
    return Database(config.database_path(cfg))


def _confirm(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


# ---------------------------------------------------------------- commands

def cmd_identify(cfg: dict, camera: int | None, web: bool) -> int:
    if web:
        from urec import web as web_mod
        web_mod.serve(cfg, camera)
    else:
        from urec import identify
        identify.run(cfg, camera)
    return 0


def cmd_register(cfg: dict, name: str, camera: int | None, force: bool) -> int:
    from urec import register
    from urec.embedder import get_embedder
    from urec.matcher import Matcher

    try:
        name = clean_name(name)
    except ValueError as exc:
        print(exc)
        return 1
    with _db(cfg) as db:
        if db.has(name) and not force and not _confirm(f"{name} is already registered. Scan again and replace?"):
            return 1
        print("Loading face model ...", flush=True)
        embedder = get_embedder(cfg)
        print(f"Scanning {name}. Follow the prompts in the camera window.")
        vectors = register.scan(cfg, embedder, name, camera)
        if vectors is None:
            return 1

        others = {n: v for n, v in db.gallery(embedder.name).items() if n.lower() != name.lower()}
        if others:
            mean = vectors.mean(axis=0)
            match, score = Matcher(others, config.threshold(cfg)).match(mean / np.linalg.norm(mean))
            if match and not force and not _confirm(
                    f"This face looks like {match} (similarity {score:.2f}). Save as {name} anyway?"):
                return 1

        db.save_person(name, embedder.name, vectors)
    print(f"Saved {name}: {len(vectors)} face samples ({embedder.name} backend).")
    return 0


def cmd_list(cfg: dict) -> int:
    backend = cfg["recognition"]["backend"]
    with _db(cfg) as db:
        people = db.people()
    if not people:
        print("Nobody is registered yet.")
        return 0
    print(f"{'Name':<24}{'Samples':<10}Registered")
    for name, created, counts in people:
        n = counts.get(backend, 0)
        note = "" if n else f"   (re-register for '{backend}')"
        print(f"{name:<24}{n:<10}{created[:10]}{note}")
    return 0


def cmd_delete(cfg: dict, name: str, yes: bool) -> int:
    with _db(cfg) as db:
        if not db.has(name):
            print(f"No one called {name!r} is registered.")
            return 1
        if not yes and not _confirm(f"Delete {name}?"):
            return 1
        db.delete(name)
    print(f"Deleted {name}.")
    return 0


def cmd_reset(cfg: dict, yes: bool) -> int:
    if not yes and input("This deletes EVERY registered person. Type YES to confirm: ").strip() != "YES":
        print("Reset cancelled.")
        return 1
    with _db(cfg) as db:
        n = db.reset()
    print(f"Database reset: removed {n} people.")
    return 0


def cmd_info(cfg: dict) -> int:
    from urec.runtime import providers

    rec = cfg["recognition"]
    with _db(cfg) as db:
        count = len(db.people())
    print(f"Config file : {config.CONFIG_FILE if config.CONFIG_FILE.exists() else 'defaults (no config.toml)'}")
    print(f"Backend     : {rec['backend']}"
          + (f" ({rec['insightface_model']})" if rec["backend"] == "insightface" else ""))
    print(f"Threshold   : {config.threshold(cfg)}")
    print(f"Body model  : {cfg['bodies']['model']}")
    print(f"Devices     : {', '.join(providers(rec['device']))}")
    print(f"Models dir  : {models.model_dir()}")
    print(f"Database    : {config.database_path(cfg)} ({count} people)")
    return 0


def cmd_download(cfg: dict) -> int:
    for key in ("yunet", "sface", cfg["bodies"]["model"]):
        print(f"{key:<12}{models.get(key)}")
    if cfg["recognition"]["backend"] == "insightface":
        from urec.embedder import get_embedder
        get_embedder(cfg)  # insightface downloads its own models on first load
        print("insightface ready")
    return 0


# ---------------------------------------------------------------- menu

MENU = """
=== U-REC ===   backend: {backend}   people: {people}
 [1] Identify (camera window)
 [2] Identify in browser (localhost)
 [3] Register a person
 [4] List people
 [5] Delete a person
 [6] Reset database
 [q] Quit
"""


def menu(cfg: dict, camera: int | None) -> int:
    while True:
        with _db(cfg) as db:
            people = len(db.people())
        print(MENU.format(backend=cfg["recognition"]["backend"], people=people))
        try:
            choice = input("Choose: ").strip().lower()
        except EOFError:  # Ctrl+D / closed input
            print()
            return 0
        try:
            if choice == "1":
                cmd_identify(cfg, camera, web=False)
            elif choice == "2":
                cmd_identify(cfg, camera, web=True)
            elif choice == "3":
                cmd_register(cfg, input("Name: "), camera, force=False)
            elif choice == "4":
                cmd_list(cfg)
            elif choice == "5":
                cmd_list(cfg)
                cmd_delete(cfg, input("Name to delete: ").strip(), yes=False)
            elif choice == "6":
                cmd_reset(cfg, yes=False)
            elif choice in ("q", "quit", "exit"):
                return 0
            else:
                print("Pick 1-6 or q.")
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as exc:
            print(f"Error: {exc}")


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
        print("Created config.toml" if config.init_file() else "config.toml already exists")
        return 0
    if args.command == "set-backend":
        config.set_backend(args.backend)
        print(f"Backend set to {args.backend} in config.toml")
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
        print("\nStopped.")
        return 130
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
