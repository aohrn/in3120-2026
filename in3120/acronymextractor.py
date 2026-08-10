# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

import re
from dataclasses import dataclass
from typing import Iterator


class AcronymExtractor:
    """
    Extracts (acronym, definition) pairs from a buffer, based on the core algorithm by Schwarz and Hearst
    described in their paper "A Simple Algorithm for Identifying Abbreviation Definitions in Biomedical Text"
    with some minor extensions.

    Uses the presence of parentheses as a simple syntactic cue to trigger the extraction and alignment logic.
    The parentheses can be used to hold either the short form (e.g., "deoxyribonucleic acid (DNA)") or the long
    form (e.g., "DNA (deoxyribonucleic acid)"). Not currently supported are language-specific cue phrases like
    "stands for" or similar (e.g., "DNA stands for deoxyribonucleic acid".)
 
    Buffers to be processed are assumed to be plain text, void of any markup. All markup such as HTML is assumed
    stripped away externally before this extraction heuristic is applied.

    The current implementation assumes English, but should work reasonably well for most Western languages. The
    Schwarz-Hearst algorithm was originally designed to extract abbreviations from well-behaved biomedical texts.
    If applied to random content from the Internet or noisy content that's not syntactially well-behaved, some
    robustness improvements are probably in order.

    The results from this extraction algorithm applied to a large corpus can be used to, e.g., build a dictionary
    to use for query expansion, or realize a service similar to https://www.acronymfinder.com/, or populate a
    searchable document field that generates a ranking boost, or something else.
    """

    @dataclass
    class Result:
        """
        An individual extraction result, as reported back to the client.
        """
        acronym: str     # The short form, e.g., "DNA".
        definition: str  # The long form, e.g., "deoxyribonucleic acid".

    def __init__(self):
        self._finder = re.compile(r"(?<=\s)\([^\)]{2,100}\)", re.UNICODE)           # Locates parenthesized stuff that triggers attempted alignment.
        self._tail   = re.compile(r"\b(\w+)\b$", re.UNICODE)                        # Extracts the last word in the buffer.
        self._fluff  = re.compile(r'\b(and|a[st]|[io]n|of|for|the)\b', re.UNICODE)  # Fluff words that should not start a definition.
        self._spaces = re.compile(r"\s+", re.UNICODE)                               # Find sequences of whitespace characters.
        self._ignore = {"-", "&"}                                                   # Characters to ignore in acronyms wrt alignment.

    def _last(self, buffer: str) -> str:
        """
        Returns the last word in the buffer.
        """
        last = self._tail.search(buffer)
        return last.group(1) if last else ""

    def _align(self, short: str, long: str) -> Result | None:
        """
        Implements the Schwarz-Hearst alignment algorithm, with some minor extensions.
        Returns None if the alignment was not successful.

        Deviations from the published Schwarz-Hearst alignment algorithm include:

        - Special-casing of some symbols, e.g., hyphens. Allowed in acronyms, but ignored for alignment.
        - The ability to "undo" the alignment of a symbol, if it matches the start of a fluff word.
        - The final sanity-check of the aligned description, e.g., to avoid starting with something hyphenated.
        """
        i = len(long) - 1
        j = len(short) - 1
        n = 2

        # Align, if possible.
        while j >= 0:
            current = short[j].lower()
            if current not in self._ignore:
                if not current.isalnum():
                    return None
                while (i >= 0 and long[i].lower() != current) or (j == 0 and i > 0 and long[i - 1].isalnum()):
                    i -= 1
                if i < 0:
                    return None
                if n > 0 and self._fluff.match(long, i):
                    n -= 1
                    j += 1
                if i > 0:
                    i -= 1
            j -= 1

        # Basic alignment succeeded. Do a final sanity-check.
        reject = i > 0 and not long[i].isspace()
        return None if reject else self.Result(short, long[i:].strip())

    def _suppress(self, result: Result) -> bool:
        """
        Returns True iff the result looks fishy and should be suppressed.
        """
        acronym, definition = result.acronym, result.definition
        return (
            not acronym or
            not definition or
            not (2 <= len(acronym) <= 10) or
            not acronym[0].isalpha() or
            not any(c.isupper() for c in acronym) or
            len(definition) <= len(acronym) or
            acronym.lower() in definition.lower().split() or
            len(definition.split()) > min(len(acronym) + 5, len(acronym) * 2)
        )

    def extract(self, buffer: str) -> Iterator[Result]:
        """
        Extracts matches from the input buffer, and returns these.
        """
        for candidate in self._finder.finditer(buffer):
            inside = buffer[(candidate.start() + 1):(candidate.end() - 1)].strip()
            inside = self._spaces.sub(" ", inside)
            before = buffer[max(0, candidate.start() - 100):candidate.start()].strip()
            before = self._spaces.sub(" ", before)
            if not before:
                continue
            if any(c.isspace() for c in inside):
                short, long = self._last(before), inside
            else:
                short, long = inside, before
            result = self._align(short, long)
            if result and not self._suppress(result):
                yield result
