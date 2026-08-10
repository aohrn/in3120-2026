# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Tuple
from .analyzer import Span, Analyzer


class WindowFinder:
    """
    Finds the smallest window in a text buffer that contains all the query terms, as measured by the number
    of terms in the window. The width of the window can be a useful engineered feature when learning to rank,
    and the window itself (possibly slightly padded for additional context) can be used as a result snippet.
    Result snippets are sometimes also called dynamic summaries, keyword-in-context snippets, or dynamic teasers.

    For details and background, see https://nlp.stanford.edu/IR-book/html/htmledition/query-term-proximity-1.html,
    https://nlp.stanford.edu/IR-book/html/htmledition/a-simple-example-of-machine-learned-scoring-1.html#sec:mls,
    and https://nlp.stanford.edu/IR-book/html/htmledition/results-snippets-1.html.

    Note that finding the smallest window is just one factor that goes into a production-ready snippet algorithm. For
    example, we might not assign equal emphasis to each query term (think TF-IDF statistics), we might not require that
    the window contains all query terms (think "enough" high-value terms), and we might want to reward windows that
    preserve the original ordering of query terms as much as possible (think term-level edit distance). As such,
    the size of the window just becomes one out of several other factors to combine in finding an "optimal" window.
    """

    @dataclass
    class Result:
        """
        A 'window' as produced by the window finder algorithm. The width equals the value ω when the window is minimal.
        The character-level window offsets are relative to the input buffer, and can be used by the client to slice the
        input buffer to yield a plausible result snippet.
        """
        begin: int  # Where in the buffer the window begins, character-level offset.
        end: int    # Where in the buffer the window ends, character-level offset.
        width: int  # How many tokens that are contained in the window.

    def __init__(self, analyzer: Analyzer):
        self._analyzer = analyzer

    def scan(self, buffer: str, query: str) -> None | Result:
        """
        Finds the smallest window in a text buffer that contains all the query terms, as measured by the number
        of terms in the window. The number of terms in such a minimal window is denoted as ω in the textbook.

        This demonstration implementation works on the level of strings and text buffers. A much more efficient
        implementation might instead work on the level of the positional postings data in a positional index. That way,
        we would just be working with the positions of the query terms in the buffer instead of the positions of all
        terms in the buffer, and we would not have to re-tokenize and re-normalize the text buffer. The algorithm
        fundamentals would be similar, though.

        The current implementation uses a sliding window approach to solve the "minimum window substring" problem
        that is sometimes used as an online technical interview question (Google it), but generalized to work on
        query terms instead of characters and slightly improved.

        Currently, a single window is returned. We can easily generalize the implementation to return multiple
        windows, e.g., in the case where there are ties or multiple windows that are "almost as small" as the
        smallest one.

        If no minimal window can be found, None is returned. The implementation assumes that the query string and
        the buffer have both been canonicalized beforehand, so that the returned window offsets are correctly
        interpreted.
        """
        # The individual normalized query terms, including their counts if they are not distinct.
        query_terms = Counter(term for term, _ in self._analyzer.terms(query))
        required_terms = len(query_terms)

        # Bookkeeping.
        sliding_window: Deque[Tuple[str, Span]] = deque()  # Our sliding window over the buffer.
        term_counts: Counter[str] = Counter()              # Our current distribution of query terms within the sliding window.
        covered_terms = 0                                  # How many of the query terms that we have fully covered within the sliding window.
        best_window = None                                 # The best window found so far.

        # Scan, keeping a sliding window!
        for buffer_term, span in self._analyzer.terms(buffer):

            # If our window is empty and would still continue to be empty, just skip it.
            is_query_term = buffer_term in query_terms
            if covered_terms == 0 and not is_query_term:
                continue

            # Grow our sliding window on the right. Update our window statistics, if needed.
            sliding_window.append((buffer_term, span))
            if is_query_term:
                term_counts[buffer_term] += 1
                if term_counts[buffer_term] == query_terms[buffer_term]:
                    covered_terms += 1

            # If our window now contains everything we need, try to shrink it.
            while covered_terms == required_terms:

                # Smallest window found so far?
                if best_window is None or len(sliding_window) < best_window.width:
                    best_window = self.Result(begin=sliding_window[0][1][0], end=sliding_window[-1][1][1], width=len(sliding_window))

                # Shrink the window from the left. Update our window statistics, if needed.
                leftmost_term, _ = sliding_window.popleft()
                if leftmost_term in query_terms:
                    term_counts[leftmost_term] -= 1
                    if term_counts[leftmost_term] < query_terms[leftmost_term]:
                        covered_terms -= 1

        # Emit results, if any.
        return best_window
