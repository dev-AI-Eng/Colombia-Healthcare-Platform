"""Parse one clinic file in a throwaway process, and write the result to stdout.

This module is the child side of `reader.read_isolated`. It exists because the
parsing libraries are the largest attack surface the importer has: a clinic file
arrives from outside, and `zipfile`, the XML parser and openpyxl are all C or
C-adjacent code reading attacker-shaped input. The archive guards in `reader`
refuse the attacks we know about — a zip bomb, an external entity, a renamed
executable — but a memory-exhaustion or segfault bug in a library is not
something a guard in our code can catch.

Running the parse here means the worst case is a dead child process and a
refusal the reviewer can read, rather than a dead API worker.

The protocol is deliberately dull: arguments in `argv`, a pickled `ReadResult`
on stdout, diagnostics on stderr. Pickle is safe in this direction because the
parent trusts the child it spawned; nothing unpickles data from a clinic file.

Run as `python -m src.onboarding.parse_worker <path> [answers-json]`.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

#: Distinguishes a refusal the reader decided on from a crash. The parent reads
#: the exit code, so a refusal is not reported as a parser failure and a parser
#: failure is not reported as a bad file.
EXIT_UNREADABLE = 3


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print("usage: parse_worker <path> [answers-json]", file=sys.stderr)
        return 2

    # Imported here, not at module scope: an import error should be reported by
    # the parent as a failed parse rather than crashing before argv is checked.
    from src.onboarding.reader import UnreadableFile, read

    path = Path(argv[0])
    answers = json.loads(argv[1]) if len(argv) == 2 else None

    try:
        result = read(path, answers)
    except UnreadableFile as refusal:
        # A decision, not a failure. The message is written for the person who
        # uploaded the file, so it is passed through verbatim.
        print(str(refusal), file=sys.stderr)
        return EXIT_UNREADABLE

    # stdout is binary here; anything a library printed to stdout during parsing
    # would corrupt the stream, so the parent only ever reads what follows.
    sys.stdout.buffer.write(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    sys.exit(main(sys.argv[1:]))
