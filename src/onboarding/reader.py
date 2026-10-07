"""Turn an uploaded clinic file into rows of strings, or refuse it.

Three rules govern this module:

**Nothing is inferred.** Values are read as text exactly as stored. Type
inference is what silently turns a cédula into a float and a Colombian date into
an American one, so conversion happens later, in `normalizers`, driven by the
mapping a human confirmed. CLAUDE.md §11 names inferring transforms from sampled
rows as the failure that sank the client's previous project.

**Refusal beats a guess.** Where a file is genuinely ambiguous, the reader
raises rather than picking. A loud failure costs a re-export; a wrong guess
writes a wrong patient record that nothing downstream can detect.

**The file is hostile until proven otherwise.** It arrives from outside, so the
archive is inspected before it is opened, and `run_isolated` exists to run the
whole parse in a process that can be killed.
"""

from __future__ import annotations

import codecs
import csv
import io
import json
import pickle
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

# defusedxml must be imported before openpyxl parses anything: openpyxl's own
# documentation states it does not defend against billion-laughs or quadratic
# blowup unless defusedxml is installed. Importing it patches the XML stack.
import defusedxml  # noqa: F401  (imported for the side effect, not the name)

if TYPE_CHECKING:
    from openpyxl.worksheet.worksheet import Worksheet

# Signatures we accept, checked against the bytes rather than the extension.
_ZIP_MAGIC: Final = b"PK\x03\x04"
_OLE2_MAGIC: Final = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls

# Archive limits. A crafted .xlsx can be a few kilobytes and expand to gigabytes.
MAX_UNCOMPRESSED_BYTES: Final = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO: Final = 200
MAX_ARCHIVE_MEMBERS: Final = 2_000

# One CSV field. The stdlib default of 128 KiB is smaller than a pasted clinical
# note, and a field that long is unusual but not an attack: the whole file is
# already bounded by the upload limit.
MAX_CSV_FIELD_BYTES: Final = 8 * 1024 * 1024

# Encoding ladder, strictest first. A probabilistic detector is deliberately not
# used: guessing wrong corrupts every accented name in the file, and Spanish
# clinic data is nothing but accented names.
ENCODING_LADDER: Final = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# Byte-order marks that identify an encoding outright. These are checked before
# the ladder because `cp1252` and `latin-1` decode UTF-16 bytes "successfully"
# into text full of NUL characters: the ladder cannot fail its way past them, so
# a UTF-16 file would be read as mojibake and imported as garbage. Excel writes
# UTF-16LE whenever a user picks "Unicode Text (*.txt)".
_BOMS: Final = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# Delimiters Excel actually writes. Semicolon is the default in Spanish locales,
# where the comma is the decimal separator.
_CANDIDATE_DELIMITERS: Final = (";", ",", "\t", "|")

# Returned when no candidate delimiter splits the file. A one-column list of
# document numbers is a legitimate import, so it is read as a single column
# rather than refused. ASCII record separator does not occur in spreadsheet
# text, so no line is ever split on it.
SINGLE_COLUMN: Final = chr(30)


#: `preexec_fn` and RLIMIT_AS are POSIX only. On Windows the timeout and the
#: process boundary still apply; the memory cap does not. That is stated rather
#: than worked around, because a Windows job object would be a new dependency for
#: a limit the archive guards already approximate.
try:
    import resource

    _CAN_LIMIT = True
except ImportError:  # pragma: no cover - Windows
    _CAN_LIMIT = False

#: Where the worker runs from, so `-m` resolves the package however the API was
#: started.
_PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]

#: Mirrors `parse_worker.EXIT_UNREADABLE`. Not imported from there: the worker
#: imports this module, and an import cycle would be worse than one duplicated
#: integer with a test asserting the two agree.
EXIT_UNREADABLE: Final = 3

#: How long a parse may take before the child is killed. Generous, because a
#: 5,000-row workbook on a slow disk is a normal import. What this stops is a
#: parse that never finishes holding an API worker forever.
PARSE_TIMEOUT_SECONDS: Final = 120

#: Address space the child may use, where the platform enforces it. openpyxl
#: holds a sheet in memory, so this exceeds the largest real workbook; what it
#: stops is a decompression bug allocating without bound.
PARSE_MEMORY_BYTES: Final = 2 * 1024 * 1024 * 1024


#: Questions whose unanswered state means the "headings" may be patient data
#: rather than column names: either row 1 is the data, or the real headings are
#: somewhere below a title line. Nothing may be sent to a model about a column
#: whose name might be a cédula.
#:
#: `csv.duplicate_headers` and `csv.overlong_rows` are deliberately absent: both
#: describe a file whose headings are genuine, so a model may be asked about
#: them. Adding a question id here is a decision about what leaves the machine.
HEADINGS_MAY_BE_DATA: Final = frozenset({"csv.no_header_row", "csv.header_row"})


class UnreadableFile(Exception):
    """The file cannot be read safely, or cannot be read without guessing."""


@dataclass(frozen=True, slots=True)
class StructureQuestion:
    """Something about the file's shape that the file cannot settle by itself.

    A warning tells a reviewer what happened. A question asks them what should
    happen, and nothing is imported until they answer. The difference is whether
    a wrong choice loses data: reading a title line as a header loses every
    patient in the file, so it is a question, while a Windows-1252 encoding is
    reported and read.

    `applied_if_approved` is what we do on approval. Declining always means the
    reading we would have used anyway, so declining is never destructive, and a
    reviewer who does not understand the question is safe either way.
    """

    #: Stable identifier, so an answer survives a re-read of the same file.
    id: str
    #: What was found, in a receptionist's terms.
    finding: str
    #: What approving does.
    applied_if_approved: str
    #: What declining does.
    applied_if_declined: str
    #: True approved, False declined, None not yet answered. A question stays
    #: listed once answered, so a reviewer can see what they decided; this is
    #: what distinguishes "still open" from "still shown".
    answered: bool | None = None


@dataclass(frozen=True, slots=True)
class Sheet:
    """One table of strings, with the structure findings that produced it."""

    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    #: 1-based row in the source file where `headers` was found.
    header_row: int
    #: The source-file row each entry of `rows` came from, in the same order.
    #: Blank rows are dropped while reading, so position in `rows` is not the
    #: row a reviewer sees: without this, a flag meant for a hidden row lands on
    #: whichever row happens to sit at that offset. Empty means "numbered from
    #: `header_row` + 1", which is the old behaviour and is right when no row
    #: was dropped.
    row_numbers: tuple[int, ...] = ()
    #: Rows the file marked hidden. Imported, but flagged: hiding a row is not
    #: deleting it, and dropping it silently would lose a real patient.
    hidden_row_numbers: tuple[int, ...] = ()
    #: Columns the file marked hidden; often the clinic's real internal code.
    hidden_columns: tuple[str, ...] = ()
    #: Findings a human should see before confirming the mapping.
    warnings: tuple[str, ...] = ()
    #: Findings a human must decide before anything is imported.
    questions: tuple[StructureQuestion, ...] = ()

    @property
    def headings_are_settled(self) -> bool:
        """Whether `headers` can be trusted to be column names.

        Used to decide whether a column name may be sent to a model: a heading
        the clinic wrote is not patient data, and a cédula read as a heading is.

        A question about which row holds the headings leaves them unsettled
        while it is open, and **declining it does not settle them**: declining
        means "read row 1 as the headings", which is the reviewer asserting what
        the file could not show. They may be right, but the cost of being wrong
        is a patient's cédula leaving the machine, and the dictionary and fuzzy
        stages still run on those columns either way. Approving does settle
        them, because the headings are then positional names we generated.
        """
        return not any(
            q.id in HEADINGS_MAY_BE_DATA and q.answered is not True for q in self.questions
        )


@dataclass(frozen=True, slots=True)
class ReadResult:
    sheets: tuple[Sheet, ...]
    encoding: str | None = None
    delimiter: str | None = None
    warnings: tuple[str, ...] = field(default=())

    @property
    def questions(self) -> tuple[StructureQuestion, ...]:
        """Every sheet's structure questions, in order."""
        return tuple(q for sheet in self.sheets for q in sheet.questions)


# --------------------------------------------------------------------- guards
def inspect_archive(path: Path) -> None:
    """Refuse an archive that would expand far beyond its size on disk.

    The declared `file_size` in a ZIP's central directory is attacker-controlled,
    so it is treated as a claim to check rather than a fact, and the real
    expansion is measured by decompressing with a cap.
    """
    try:
        archive_context = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise UnreadableFile(
            "File starts like a spreadsheet but is not a valid archive; it may be truncated."
        ) from error

    with archive_context as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise UnreadableFile(
                f"Archive declares {len(members)} members, more than the {MAX_ARCHIVE_MEMBERS} allowed."
            )

        declared = sum(member.file_size for member in members)
        compressed = sum(member.compress_size for member in members) or 1
        if declared > MAX_UNCOMPRESSED_BYTES:
            raise UnreadableFile(
                f"Archive declares {declared:,} bytes uncompressed, "
                f"more than the {MAX_UNCOMPRESSED_BYTES:,} allowed."
            )
        if declared / compressed > MAX_COMPRESSION_RATIO:
            raise UnreadableFile(
                f"Archive expands {declared / compressed:.0f}x, "
                f"more than the {MAX_COMPRESSION_RATIO}x allowed."
            )

        # The declaration may lie, so read the members and count what arrives.
        total = 0
        for member in members:
            with archive.open(member) as stream:
                while chunk := stream.read(64 * 1024):
                    total += len(chunk)
                    if total > MAX_UNCOMPRESSED_BYTES:
                        raise UnreadableFile(
                            f"Archive expands past the {MAX_UNCOMPRESSED_BYTES:,} byte limit; "
                            "its declared sizes understate it."
                        )


def detect_kind(path: Path) -> str:
    """Return 'xlsx', 'xls' or 'csv' from the file's bytes, not its name."""
    head = path.read_bytes()[:8]
    if head.startswith(_ZIP_MAGIC):
        return "xlsx"
    if head.startswith(_OLE2_MAGIC):
        return "xls"
    # Anything else must be text to be a CSV. An executable renamed .csv fails here.
    if head[:2] == b"MZ" or head[:4] == b"\x7fELF":
        raise UnreadableFile("File is an executable, not a spreadsheet.")
    return "csv"


# ----------------------------------------------------------------------- CSV
def decode(raw: bytes) -> tuple[str, str]:
    """Decode with the strictest encoding that accepts the whole file.

    Returns the text and the encoding used. `latin-1` cannot fail, so it is the
    floor rather than a real detection; when it is reached the caller is warned,
    because a wrong decoding shows up as mojibake in a patient's name.

    A byte-order mark is trusted over the ladder. UTF-32's mark begins with
    UTF-16LE's, so the wider marks are tested first.
    """

    # A UTF-16 file with no BOM is half NUL bytes, and every single-byte codec
    # "succeeds" on it: the ladder cannot fail its way past cp1252, so the file
    # would be read as text full of NULs and imported as garbage. Excel's own
    # UTF-16 export carries a BOM and is handled by the loop below, which runs
    # first so a BOM is always consumed by a codec that strips it; a
    # programmatic export may have no BOM at all.
    #
    # Byte order comes from where the NULs sit, not from whether decoding
    # raises: UTF-16LE reading of big-endian bytes yields CJK characters rather
    # than failing, so "it decoded" is not evidence of the right order. In
    # ASCII-ish text LE puts its NULs at odd offsets and BE at even ones.
    def _bomless_utf16(raw: bytes) -> tuple[str, str] | None:
        probe = raw[:4096]
        if len(probe) < 4 or probe.count(0) <= len(probe) // 4:
            return None
        even = sum(1 for i in range(0, len(probe) - 1, 2) if probe[i] == 0)
        odd = sum(1 for i in range(1, len(probe), 2) if probe[i] == 0)
        if even == odd:
            return None  # no clear order; fall through to the ladder
        encoding = "utf-16-be" if even > odd else "utf-16-le"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            return None
        if "\x00" in text:
            return None  # not actually UTF-16 text, whatever it is
        return text, encoding

    for mark, encoding in _BOMS:
        if raw.startswith(mark):
            try:
                # `utf-16`/`utf-32` consume the mark themselves and pick the
                # byte order from it, so the prefix is not stripped by hand.
                return raw.decode(encoding), encoding
            except UnicodeDecodeError as error:
                raise UnreadableFile(
                    f"File begins with a {encoding} byte-order mark but is not valid "
                    f"{encoding} ({error.reason})."
                ) from error

    if (found := _bomless_utf16(raw)) is not None:
        return found

    for encoding in ENCODING_LADDER:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnreadableFile("File is not decodable as text in any supported encoding.")


def sniff_delimiter(sample: str) -> str:
    """Score the candidate delimiters instead of asking `csv.Sniffer`.

    The rule is consistency: the right delimiter splits every line into the same
    number of fields. `csv.Sniffer` guesses from character frequency and reads
    `1.250,50` as a comma-separated pair, which silently splits a Spanish
    currency column in two.
    """
    lines = [line for line in sample.splitlines() if line.strip()][:20]
    if not lines:
        raise UnreadableFile("File is empty.")

    scored: list[tuple[float, int, str]] = []  # (agreement, width, delimiter)
    for delimiter in _CANDIDATE_DELIMITERS:
        try:
            rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
        except csv.Error:
            continue
        widths = [len(row) for row in rows if row]
        if not widths:
            continue
        # The most frequent width, and among ties the widest: a delimiter must
        # not be rewarded for leaving most lines unsplit.
        top = max(widths.count(width) for width in set(widths))
        width = max(w for w in set(widths) if widths.count(w) == top)
        if width < 2:
            continue  # this delimiter does not split the file at all
        # Agreement, not frequency. An address column full of commas splits some
        # lines and not others, and that disagreement is what rules the comma
        # out even where it yields as many fields as the real delimiter.
        scored.append((top / len(widths), width, delimiter))

    if not scored:
        # A one-column file is a legitimate import (a list of document numbers,
        # say), so it is read as a single column rather than refused. The
        # delimiter is one that cannot occur in text, so nothing is split.
        return SINGLE_COLUMN

    # Ranked on agreement first, then width. Both are properties of the file, so
    # the answer does not depend on the order the candidates were tried in.
    return max(scored)[2]


def _csv_rows(text: str, delimiter: str, *, quoted: bool = True) -> list[list[str]]:
    """Parse with a field-size limit wide enough for a pasted clinical note.

    `csv` refuses a field over 128 KiB, and a receptionist pasting a long note
    into one cell is enough to hit it. The limit is raised for this parse only
    and restored afterwards, because it is process-global state and leaving it
    raised would weaken the guard for every other caller.

    With `quoted=False` a quote character is just a character, which is how a
    file holding one stray `"` is recovered: the default reading swallows every
    line after it into a single cell.
    """
    previous = csv.field_size_limit()
    csv.field_size_limit(MAX_CSV_FIELD_BYTES)
    try:
        reader = (
            csv.reader(io.StringIO(text), delimiter=delimiter)
            if quoted
            else csv.reader(io.StringIO(text), delimiter=delimiter, quoting=csv.QUOTE_NONE)
        )
        return list(reader)
    finally:
        csv.field_size_limit(previous)


def _find_csv_header(rows: list[tuple[str, ...]]) -> int:
    """Index of the row that most looks like column labels, within the first few.

    A hand-kept CSV often opens with a clinic name or a date, and taking row 1 on
    faith reads that title as the only column and every real row as ragged. The
    same scorer the Excel reader uses decides it, so the two agree.

    Rows are only considered while they could plausibly be a preamble: a header
    twenty rows down is a different problem, and guessing there would be worse
    than asking.
    """
    best, best_score = 0, -1.0
    for index, row in enumerate(rows[:10]):
        score = _header_score(list(row))
        # Ties go to the earlier row: with two plausible headers the first table
        # is the one the file is about.
        if score > best_score:
            best, best_score = index, score
    # Nothing here looks like column labels. Returning the least-bad row would
    # claim a preamble the file does not have and discard the patients above it
    # -- a positional export such as the RIPS archivo has no headings anywhere,
    # and every row scores low. Reporting 0 says "no preamble found", which is
    # what lets the headerless question be asked instead.
    if best_score < HEADER_SCORE_FLOOR:
        return 0
    return best


#: A row scoring below this does not look like column labels.
#:
#: `_header_score` sums four ratios, so a row that is entirely distinct,
#: entirely textual, entirely short and fully populated scores exactly 4.0 —
#: which every real heading row does. A data row loses points on at least one:
#: "CC;1020304050;Ana Perez" scores 3.67 because the cédula is not textual, and
#: a row of numbers scores 3.0. Measured across both sets, the gap is 3.67 to
#: 4.00, so the floor sits just under 4.
#:
#: This decides only whether to **ask**, never what the answer is. A heading row
#: and a data row can be genuinely identical in shape — a clinic whose columns
#: are all words, with no heading — and nothing in the file resolves that, so a
#: person does.
HEADER_SCORE_FLOOR: Final = 3.9


def _looks_like_a_header(row: tuple[str, ...]) -> bool:
    """Whether this row can be taken for column labels at all.

    A one-column file is exempt: `_header_score` returns 0 for fewer than two
    filled cells, so scoring it would flag every such file. There is also
    nothing useful to ask -- a single column of document numbers with no
    heading looks exactly like one with a heading, and positional naming would
    not help a reviewer who can see the column.
    """
    filled = [cell for cell in row if cell.strip()]
    if len(filled) < 2:
        return True
    # Scored on the distinct values. A repeated heading ("NOMBRE;NOMBRE;TEL")
    # costs distinctness and would otherwise read as "not a heading row", which
    # is a second question for something `csv.duplicate_headers` already
    # explains -- and a reviewer who answers the wrong one first gets a worse
    # file. Repetition is a naming problem, not evidence about the shape.
    return _header_score(list(dict.fromkeys(filled))) >= HEADER_SCORE_FLOOR


def read_csv(path: Path, answers: Mapping[str, bool] | None = None) -> ReadResult:
    """Read a delimited file.

    `answers` carries the reviewer's decisions on this file's structure
    questions, keyed by question id. An unanswered question is still asked and
    the declined (non-destructive) reading is used meanwhile, so a file is always
    readable and never silently altered.
    """
    raw = path.read_bytes()
    if raw[:2] == b"MZ":
        raise UnreadableFile("File is an executable, not a spreadsheet.")

    text, encoding = decode(raw)

    # Classic Mac exports end lines with a bare CR, which `csv` treats as a
    # newline inside an unquoted field and refuses with an opaque error. There
    # is nothing to decide here: a CR that is not part of CRLF is a line ending,
    # so it is normalised rather than reported.
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    warnings: list[str] = []
    if encoding == "latin-1":
        warnings.append("Encoding could not be determined; read as latin-1. Check accented names.")
    elif encoding == "cp1252":
        warnings.append("Read as Windows-1252 (a Spanish Excel export). Check accented names.")

    decided = dict(answers or {})
    delimiter = sniff_delimiter(text[:64_000])
    # A cell that opens a quote and never closes it swallows every following
    # line into itself, so three patients arrive as one row. The field-count
    # warning that follows blames the row shape, which sends a reviewer to the
    # wrong place, and nothing says the rows are gone. Reading the file again
    # with quoting off recovers them, so the reviewer is asked which reading is
    # right. Declining keeps the quoted reading, which is what we would have
    # done unasked.
    unbalanced = text.count('"') % 2 == 1
    reparse_literally = unbalanced and bool(decided.get("csv.unbalanced_quote"))
    try:
        rows = [
            tuple(cell.strip() for cell in row)
            for row in _csv_rows(text, delimiter, quoted=not reparse_literally)
        ]
    except csv.Error as error:
        # Every csv.Error we can predict is handled above, so reaching here means
        # the file is malformed in a way we have not seen. It is reported as
        # unreadable rather than escaping as a 500.
        raise UnreadableFile(f"File is not valid CSV: {error}") from error
    # Blank lines are dropped, but the line each surviving row came from is kept:
    # a row number is how a refusal points a receptionist at the line to fix, and
    # it is what `excluded_rows` names, so it has to mean the line in their file.
    numbered = [
        (number, row) for number, row in enumerate(rows, start=1) if any(cell for cell in row)
    ]
    if not numbered:
        raise UnreadableFile("File contains no rows.")
    line_numbers = [number for number, _ in numbered]
    rows = [row for _, row in numbered]

    questions: list[StructureQuestion] = []
    if unbalanced:
        swallowed = sum(cell.count("\n") for row in rows for cell in row)
        questions.append(
            StructureQuestion(
                id="csv.unbalanced_quote",
                answered=decided.get("csv.unbalanced_quote"),
                finding=(
                    f'The file has an odd number of " characters, and {swallowed} '
                    f"line(s) were read as part of a cell rather than as rows of "
                    f"their own. A quote opened somewhere and was never closed."
                ),
                applied_if_approved=(
                    'Read " as an ordinary character, recovering those lines as rows.'
                ),
                applied_if_declined=(
                    "Keep the quoted reading, so those lines stay inside one cell."
                ),
            )
        )

    # A preamble line above the table: taking row 1 on faith would read a clinic
    # name as the only column and lose every patient. Which row is the header is
    # a judgement, so it is asked rather than assumed.
    detected_header = _find_csv_header(rows)

    # A file whose very first row is data, with no headings anywhere, would have
    # that record silently consumed as column names. Nothing in the file says
    # which it is -- a clinic exporting without headers and one whose headings
    # happen to be numeric look identical -- so it is asked rather than guessed.
    headerless = detected_header == 0 and not _looks_like_a_header(rows[0])
    # Declining means row 1, which is the file as written.
    header_index = detected_header if decided.get("csv.header_row") else 0
    if headerless:
        questions.append(
            StructureQuestion(
                id="csv.no_header_row",
                answered=decided.get("csv.no_header_row"),
                finding=(
                    f"The first row does not look like column headings: "
                    f"{list(rows[0])[:4]!r}. If this file has no heading row, "
                    f"reading it as one would lose that record."
                ),
                applied_if_approved=(
                    "Treat the first row as data and name the columns by position "
                    "(column 1, column 2, ...), so nothing is lost."
                ),
                applied_if_declined=(
                    "Treat the first row as the headings, exactly as the file has it."
                ),
            )
        )

    if detected_header > 0:
        questions.append(
            StructureQuestion(
                id="csv.header_row",
                answered=decided.get("csv.header_row"),
                finding=(
                    f"The file opens with {detected_header} line(s) above the table: "
                    f"{rows[0][0][:60]!r}. Row {detected_header + 1} looks like the "
                    f"column headings."
                ),
                applied_if_approved=(
                    f"Use row {detected_header + 1} as the headings and ignore the line(s) above it."
                ),
                applied_if_declined="Use row 1 as the headings, exactly as the file has it.",
            )
        )

    if headerless and decided.get("csv.no_header_row"):
        # Positional names, so every row stays data. The reviewer maps them on
        # the confirmation screen exactly as they would any other column.
        headers = tuple(f"column {position}" for position in range(1, len(rows[0]) + 1))
        synthetic_header = True
    else:
        headers = rows[header_index]
        synthetic_header = False
    width = len(headers)

    # Two columns with one name: whichever is mapped, the other silently loses.
    duplicated = sorted({h for h in headers if h and headers.count(h) > 1})
    if duplicated and decided.get("csv.duplicate_headers"):
        # Numbered so each repeat can be mapped or skipped on its own. The first
        # keeps its name, so a mapping confirmed before the rename still applies.
        counts: dict[str, int] = {}
        renamed: list[str] = []
        for header in headers:
            if header and headers.count(header) > 1:
                counts[header] = counts.get(header, 0) + 1
                renamed.append(header if counts[header] == 1 else f"{header} ({counts[header]})")
            else:
                renamed.append(header)
        headers = tuple(renamed)
    if duplicated:
        questions.append(
            StructureQuestion(
                id="csv.duplicate_headers",
                answered=decided.get("csv.duplicate_headers"),
                finding=(
                    f"More than one column is called {duplicated!r}. Only one of each "
                    f"can be imported; the other would be silently ignored."
                ),
                applied_if_approved=(
                    "Number the repeats (NOMBRE, NOMBRE (2)) so each column can be "
                    "mapped or skipped on its own."
                ),
                applied_if_declined="Leave the names as they are and import the first of each.",
            )
        )

    # A title line above the table makes the header one column wide, so every
    # real row then looks overlong. Asking about both at once would show a
    # second, misleading question ("an unquoted delimiter") for a cause the
    # first question already covers. The header question is asked alone, and
    # once it is answered any rows still overlong are asked about then.
    header_question_pending = detected_header > 0 and "csv.header_row" not in decided

    body: list[tuple[str, ...]] = []
    body_numbers: list[int] = []
    overlong: list[int] = []
    # Declining leaves the row out entirely rather than cutting cells off it.
    keep_overlong = bool(decided.get("csv.overlong_rows"))
    # With invented headings the first row is data, not a heading.
    first_data = header_index if synthetic_header else header_index + 1
    for number, row in zip(line_numbers[first_data:], rows[first_data:], strict=True):
        if len(row) > width:
            overlong.append(number)
            if not keep_overlong:
                if not header_question_pending:
                    warnings.append(
                        f"Row {number} has {len(row)} fields, expected {width}; "
                        f"left out until the extra fields are confirmed."
                    )
                continue
        if len(row) != width:
            if not header_question_pending:
                warnings.append(f"Row {number} has {len(row)} fields, expected {width}.")
            row = row[:width] + ("",) * max(0, width - len(row))
        body.append(row)
        body_numbers.append(number)

    if overlong and not header_question_pending:
        shown = ", ".join(str(n) for n in overlong[:5])
        more = f" and {len(overlong) - 5} more" if len(overlong) > 5 else ""
        questions.append(
            StructureQuestion(
                id="csv.overlong_rows",
                answered=decided.get("csv.overlong_rows"),
                finding=(
                    f"Row(s) {shown}{more} have more fields than there are headings. "
                    f"The extra values have nowhere to go, which usually means an "
                    f"unquoted {delimiter!r} inside a value."
                ),
                applied_if_approved=(
                    "Drop the extra fields past the last heading and import the rest of those rows."
                ),
                applied_if_declined=(
                    "Leave those rows out of the import so nothing is silently cut."
                ),
            )
        )

    return ReadResult(
        sheets=(
            Sheet(
                name=path.stem,
                headers=headers,
                rows=tuple(body),
                # With invented headings there is no header line in the file,
                # so this reports 0: every data row still carries its own real
                # line number, which is what a refusal points at.
                header_row=0 if synthetic_header else line_numbers[header_index],
                row_numbers=tuple(body_numbers),
                warnings=tuple(warnings),
                questions=tuple(questions),
            ),
        ),
        encoding=encoding,
        delimiter=delimiter,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------- Excel
def _openpyxl() -> Any:
    """openpyxl, imported on first use.

    It pulls in numpy, which costs roughly half a second of interpreter startup.
    A CSV import never touches a workbook, and every parse now runs in its own
    process (`read_isolated`), so paying that at module scope would charge every
    CSV upload for a library it does not use.
    """
    import openpyxl

    return openpyxl


def _cell_text(value: object) -> str:
    """Render a cell as text without inventing a format.

    Dates and times are rendered ISO-first so a later normalizer can tell a real
    date cell from a string that merely looks like one. Floats keep full
    precision: `1.23457e+11` must stay visibly damaged rather than be rounded
    into a plausible document number.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        # 1045678901.0 is a document number Excel stored as a float.
        return str(int(value))
    return str(value).strip()


def _header_score(cells: list[str]) -> float:
    """How much a row looks like column labels rather than data."""
    filled = [cell for cell in cells if cell]
    if len(filled) < 2:
        return 0.0
    distinct = len(set(filled)) / len(filled)
    textual = sum(1 for cell in filled if not cell.replace(".", "", 1).isdigit()) / len(filled)
    short = sum(1 for cell in filled if len(cell) <= 40) / len(filled)
    density = len(filled) / max(len(cells), 1)
    return distinct + textual + short + density


def _find_header_row(worksheet: Worksheet, probe: int = 25) -> int:
    """Return the 1-based row that begins the FIRST table on the sheet.

    Hand-made workbooks put a clinic name and a print date above the table, so
    row 1 is often not the header. They also paste a second table below the
    first, and that second table frequently scores higher because it is smaller
    and tidier. Taking the best-scoring row anywhere on the sheet therefore
    skips the real data entirely, so the first row that scores well enough wins,
    not the best row overall.
    """
    scores: list[tuple[int, float, int]] = []  # (row, score, filled cells)
    for row_index, row in enumerate(
        worksheet.iter_rows(min_row=1, max_row=probe, values_only=True), start=1
    ):
        cells = [_cell_text(value) for value in row]
        filled = sum(1 for cell in cells if cell)
        scores.append((row_index, _header_score(cells), filled))

    if not scores:
        return 1
    best_score = max(score for _, score, _ in scores)
    if best_score <= 0:
        return 1

    # A header must span most of the table, which rules out a title row: merged
    # across the sheet it still reads as one filled cell. Width is measured
    # against the widest row on the sheet rather than against other candidates,
    # because a header with an unlabelled column ("NOMBRE", "", "", "EPS") is
    # narrower than the data beneath it and would otherwise lose to row 2 —
    # promoting a patient row to the header and dropping that patient.
    table_width = max(filled for _, _, filled in scores)
    for row_index, score, filled in scores:
        if score >= best_score * 0.85 and filled >= table_width * 0.5:
            return row_index
    return 1


def _formula_text(path: Path) -> dict[tuple[str, int, int], str]:
    """Formula text for cells that carry a formula but no cached result.

    `data_only=True` returns the value Excel last calculated. A workbook written
    by a script, or never opened since editing, has no cached value, so every
    formula cell reads as empty — a whole column of patient data silently blank
    with no error anywhere. Falling back to the formula's own text keeps the
    cell visible so validation can reject it, and keeps an injected payload
    intact so the export layer can neutralise it rather than lose it.
    """
    workbook = _openpyxl().load_workbook(path, read_only=False, data_only=False)
    try:
        formulas: dict[tuple[str, int, int], str] = {}
        for worksheet in workbook.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str) and cell.value.startswith("="):
                        formulas[(worksheet.title, cell.row, cell.column)] = cell.value
        return formulas
    finally:
        workbook.close()


@dataclass(frozen=True, slots=True)
class _SheetStructure:
    hidden_rows: frozenset[int]
    hidden_columns: tuple[str, ...]
    state: str


def _structure(path: Path) -> dict[str, _SheetStructure]:
    """Hidden rows, hidden columns and visibility per sheet.

    Read-only worksheets do not carry row or column dimensions, so this costs a
    second pass with a normal load. It is worth it: a hidden row holds a real
    patient, and a hidden column often holds the code the clinic actually keys
    on, so neither may be discovered only by accident.
    """
    workbook = _openpyxl().load_workbook(path, read_only=False, data_only=True)
    try:
        structure: dict[str, _SheetStructure] = {}
        for worksheet in workbook.worksheets:
            hidden_rows = frozenset(
                number for number, dimension in worksheet.row_dimensions.items() if dimension.hidden
            )
            hidden_columns = tuple(
                letter
                for letter, dimension in worksheet.column_dimensions.items()
                if dimension.hidden
            )
            structure[worksheet.title] = _SheetStructure(
                hidden_rows=hidden_rows,
                hidden_columns=hidden_columns,
                state=str(worksheet.sheet_state),
            )
        return structure
    finally:
        workbook.close()


def read_excel(path: Path, answers: Mapping[str, bool] | None = None) -> ReadResult:
    """Read a workbook.

    `answers` carries the reviewer's decisions, keyed by question id, exactly as
    for a CSV: a workbook can be headerless too, and the answer changes which
    row is data.
    """
    inspect_archive(path)
    try:
        structure = _structure(path)
    except UnreadableFile:
        raise
    except Exception as error:  # openpyxl raises OSError, KeyError, zipfile errors...
        raise UnreadableFile(f"File is not a readable workbook: {error}") from error

    # data_only=True returns the cached result of a formula rather than its text.
    try:
        workbook = _openpyxl().load_workbook(path, read_only=True, data_only=True)
        formulas = _formula_text(path)
    except Exception as error:
        raise UnreadableFile(f"File is not a readable workbook: {error}") from error
    try:
        sheets: list[Sheet] = []
        warnings: list[str] = []

        for worksheet in workbook.worksheets:
            # The declared dimension is a claim by the file. openpyxl trusts it
            # in read-only mode, so a sheet declaring A1:A1 over 5,000 real rows
            # silently yields one cell. Recomputing it from the data is the
            # documented remedy.
            worksheet.reset_dimensions()

            found = structure.get(worksheet.title, _SheetStructure(frozenset(), (), "visible"))
            hidden_rows_all, hidden_columns = found.hidden_rows, found.hidden_columns
            sheet_warnings: list[str] = []
            if found.state != "visible":
                sheet_warnings.append(
                    f"Sheet '{worksheet.title}' is hidden; it may hold stale data."
                )

            header_row = _find_header_row(worksheet)
            if header_row > 1:
                sheet_warnings.append(
                    f"Header found on row {header_row}; rows above it were skipped."
                )

            rows = list(worksheet.iter_rows(min_row=header_row, values_only=True))
            if not rows:
                continue

            decided = dict(answers or {})
            headers = tuple(_cell_text(value) for value in rows[0])

            # A workbook with no heading row would otherwise have its first
            # patient read as the column names: lost from the data, and sent to
            # the mapping model as if it were a heading.
            sheet_questions: list[StructureQuestion] = []
            synthetic_header = False
            if not _looks_like_a_header(headers):
                sheet_questions.append(
                    StructureQuestion(
                        id="csv.no_header_row",
                        answered=decided.get("csv.no_header_row"),
                        finding=(
                            f"Sheet '{worksheet.title}' row {header_row} does not look "
                            f"like column headings: {list(headers)[:4]!r}. If this sheet "
                            f"has no heading row, reading it as one would lose that record."
                        ),
                        applied_if_approved=(
                            "Treat the first row as data and name the columns by "
                            "position (column 1, column 2, ...), so nothing is lost."
                        ),
                        applied_if_declined=(
                            "Treat the first row as the headings, as the sheet has it."
                        ),
                    )
                )
                if decided.get("csv.no_header_row"):
                    headers = tuple(f"column {position}" for position in range(1, len(headers) + 1))
                    synthetic_header = True

            width = len(headers)
            body: list[tuple[str, ...]] = []
            body_numbers: list[int] = []
            uncalculated = 0
            data_rows = rows if synthetic_header else rows[1:]
            first_number = header_row if synthetic_header else header_row + 1
            for row_number, row in enumerate(data_rows, start=first_number):
                cells: list[str] = []
                for column_index, value in enumerate(row[:width], start=1):
                    text = _cell_text(value)
                    if not text:
                        formula = formulas.get((worksheet.title, row_number, column_index))
                        if formula:
                            # Keep the formula visible: validation rejects it,
                            # and the export layer neutralises it.
                            text = formula
                            uncalculated += 1
                    cells.append(text)
                cells.extend("" for _ in range(width - len(cells)))
                if any(cells):
                    body.append(tuple(cells))
                    # The worksheet row this came from. A blank row between two
                    # patients is dropped here, so position in `body` stops
                    # matching the row the sheet shows.
                    body_numbers.append(row_number)
            if uncalculated:
                sheet_warnings.append(
                    f"{uncalculated} cell(s) hold a formula with no calculated value; "
                    "open the file in Excel, recalculate and save before importing."
                )

            hidden_rows = tuple(sorted(n for n in hidden_rows_all if n > header_row))
            if hidden_rows:
                sheet_warnings.append(
                    f"{len(hidden_rows)} hidden row(s) were imported and flagged for review."
                )
            if hidden_columns:
                sheet_warnings.append(f"Hidden column(s) {', '.join(hidden_columns)} contain data.")

            sheets.append(
                Sheet(
                    name=worksheet.title,
                    headers=headers,
                    rows=tuple(body),
                    header_row=0 if synthetic_header else header_row,
                    row_numbers=tuple(body_numbers),
                    hidden_row_numbers=hidden_rows,
                    hidden_columns=hidden_columns,
                    warnings=tuple(sheet_warnings),
                    questions=tuple(sheet_questions),
                )
            )
            warnings.extend(sheet_warnings)

        if not sheets:
            raise UnreadableFile("Workbook contains no readable sheet.")
        return ReadResult(sheets=tuple(sheets), warnings=tuple(warnings))
    finally:
        workbook.close()


def _log() -> Any:
    """The logger, imported on use.

    Only the failure paths below log, and they run in the parent. Importing
    structlog at module scope would cost every parse worker about 240ms of
    startup for a logger it never calls.
    """
    from src.core.logging import get_logger

    return get_logger(__name__)


def read_isolated(path: Path, answers: Mapping[str, bool] | None = None) -> ReadResult:
    """Read a file in a separate process, and refuse it if that process dies.

    The parsing libraries are the importer's largest attack surface: the file
    arrives from outside, and `zipfile`, the XML parser and openpyxl are C or
    C-adjacent code reading attacker-shaped input. The guards in this module
    refuse the attacks we know about -- a zip bomb, an external entity, a renamed
    executable -- but a memory-exhaustion or segfault bug inside a library is not
    something a guard in our own code can catch. Isolating the parse means the
    worst case is a dead child and a refusal a reviewer can read, rather than a
    dead API worker.

    There is no fallback to parsing in this process: a caller that asked for
    isolation and did not get it should be told, not quietly given the thing it
    was trying to avoid.
    """
    command = [sys.executable, "-m", "src.onboarding.parse_worker", str(path)]
    if answers:
        command.append(json.dumps(dict(answers)))

    try:
        # S603: the argv is fixed in this function and `shell` is false. The only
        # value from outside is the path, which is passed as one argument and is
        # never interpreted by a shell.
        completed = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            timeout=PARSE_TIMEOUT_SECONDS,
            cwd=_PROJECT_ROOT,
            preexec_fn=_limit_child if _CAN_LIMIT else None,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        raise UnreadableFile(
            f"Reading this file took longer than {PARSE_TIMEOUT_SECONDS} seconds and was "
            f"stopped. A spreadsheet that slow to read is usually corrupt; try opening "
            f"it in Excel and saving it again."
        ) from expired

    if completed.returncode == EXIT_UNREADABLE:
        # The child decided this, and wrote the message for the person who
        # uploaded the file, so it is passed through rather than replaced.
        raise UnreadableFile(completed.stderr.decode("utf-8", "replace").strip())

    if completed.returncode != 0:
        # The parse died rather than refusing: a crash, a kill, or the memory
        # cap. The reviewer is told the file could not be read, and the detail
        # goes to the log, because a stack trace is not a message to a
        # receptionist.
        detail = completed.stderr.decode("utf-8", "replace").strip()
        _log().error(
            "onboarding.parse_crashed",
            returncode=completed.returncode,
            detail=detail[-2000:] or None,
        )
        raise UnreadableFile(
            "This file could not be read. It may be corrupt or built in a way the "
            "reader cannot handle; try saving it again from Excel, or export it as CSV."
        )

    try:
        # S301: this is the stdout of a child we spawned ourselves, running our
        # own module. No clinic file is ever unpickled -- the file goes into the
        # child as bytes and comes back as our own dataclasses.
        result = pickle.loads(completed.stdout)  # noqa: S301
    except Exception as error:
        # A truncated stream, or a library that printed to stdout and corrupted
        # it. Either way there is no result to return.
        _log().error("onboarding.parse_unreadable_output", error=str(error))
        raise UnreadableFile("This file could not be read.") from error

    if not isinstance(result, ReadResult):  # pragma: no cover - defensive
        raise UnreadableFile("This file could not be read.")
    return result


def _limit_child() -> None:  # pragma: no cover - POSIX only, runs in the child
    """Cap the child's address space so a runaway allocation dies alone.

    Only ever called where `resource` imported, which mypy cannot see because it
    type-checks for one platform at a time.
    """
    resource.setrlimit(  # type: ignore[attr-defined]
        resource.RLIMIT_AS,  # type: ignore[attr-defined]
        (PARSE_MEMORY_BYTES, PARSE_MEMORY_BYTES),
    )


def read(path: Path, answers: Mapping[str, bool] | None = None) -> ReadResult:
    """Read any supported file into sheets of strings, or refuse it.

    `answers` resolves the structure questions a previous read raised, keyed by
    question id: True applies the fix, False keeps the file as written.
    """
    kind = detect_kind(path)
    if kind == "csv":
        return read_csv(path, answers)
    if kind == "xls":
        raise UnreadableFile(
            "Legacy .xls is not supported. Save the file as .xlsx and upload it again."
        )
    return read_excel(path, answers)
