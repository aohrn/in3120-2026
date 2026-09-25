# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from dataclasses import dataclass
from typing import Iterator, Any, List
from .analyzer import Analyzer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications (we ignore the
    part about failure transitions) and some minor NLP extensions. The running time of this algorithm is in
    practice virtually insensitive to the size of the dictionary, and linear in the length of the buffer we
    are searching in.

    The analyzer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    @dataclass
    class State:
        """
        A currently explored state, as the scan proceeds.
        """
        node: Trie  # The current position in the trie, after having consumed zero or more characters.
        begin: int  # The index into the original buffer where the state was "born".
        match: str  # The symbols consumed so far to get to the current state.

    @dataclass
    class Result:
        """
        An individual result of the scan, as reported back to the client.
        """
        match: str        # The matching dictionary entry.
        meta: None | Any  # Optional mata data associated with the match, if present in the dictionary.
        surface: str      # The part of the input buffer that triggered the match, space-normalized.
        begin: int        # The index into the original buffer where the surface form starts.
        end: int          # The index into the original buffer where the surface form ends.

    def __init__(self, trie: Trie, analyzer: Analyzer):
        self._trie = trie          # The set of strings we want to detect in the scanned buffer.
        self._analyzer = analyzer  # The same that was used when the trie was built.

    def scan(self, buffer: str) -> Iterator[Result]:
        """
        Scans the given buffer once and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """
        # Break up the analyzer, to report back spans relative to the input buffer which may
        # or may not be canonicalized.
        normalizer = self._analyzer.normalizer
        tokenizer = self._analyzer.tokenizer

        # The set of currently explored states. The trie node is what we advance along the way,
        # the index is needed so that we know where we first started if/when a match is found,
        # and the match is needed so that we can differentiate between the surface form of the
        # match and the (possibly heavily normalized) base form of the match.
        states: List[StringFinder.State] = []

        # Where did the previous token end? Assume that tokens are produced sorted in left-to-right
        # order.
        previous_end = -1

        # Only consider matches that start on token boundaries.
        for string, (begin, end) in tokenizer.tokens(buffer):

            # Mirror how the trie was built, ensuring we compare apples to apples.
            # Canonicalize on a per token basis instead of doing the whole buffer upfront,
            # to ensure that offsets are retained and the ranges we report back make
            # sense to the client.
            string = normalizer.normalize(normalizer.canonicalize(string))

            # Is this token "connected to" the previous token, in the sense of the two being
            # crammed together with nothing separating them? Some languages, e.g., Japanese or
            # Chinese, don't use whitespace between tokens.
            is_connected, previous_end = (previous_end > 0) and (begin == previous_end), end

            # Inject a space for the currently live states, if needed. Prune away states that
            # don't survive.
            if not is_connected:
                states = [self.State(child, state.begin, state.match + " ") for state in states if (child := state.node.consume(" "))]

            # Consider this token a potential start for a match.
            states.append(self.State(self._trie, begin, ""))

            # Advance all currently live states with the current (normalized) token. Prune away
            # states that don't survive.
            states = [self.State(child, state.begin, state.match + string) for state in states if (child := state.node.consume(string))]

            # Report matches, if any, that end on the token we just consumed. Use the
            # tokenizer to possibly space-normalize the surface form we emit. If the client
            # requires the exact surface form and its location in the input buffer, they can
            # do that using the returned span.
            yield from (self.Result(state.match, state.node.get_meta(),
                                    tokenizer.join(tokenizer.tokens(buffer[state.begin:end])),
                                    state.begin, end) for state in states if state.node.is_final())
