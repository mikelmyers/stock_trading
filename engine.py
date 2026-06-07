"""Legacy entry point — use cli.py instead."""

import sys
from cli import main

if __name__ == "__main__":
    # Map old positional args to new subcommands
    if len(sys.argv) > 1 and sys.argv[1] in ("scan", "monitor", "full"):
        sys.exit(main(sys.argv[1:]))
    sys.exit(main(["full"] + sys.argv[1:]))