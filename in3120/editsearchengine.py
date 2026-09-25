# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=fixme
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

import math
from dataclasses import dataclass
from typing import Iterator, Iterable, Dict, Any, Callable
from .edittable import EditTable
from .analyzer import Analyzer, DummyAnalyzer
from .sieve import Sieve
from .trie import SimpleTrie, Trie


class EditSearchEngine:
    """
    Realizes a simple character-level edit distance lookup engine, that, given a larger set of strings
    encoded in a trie, finds all strings in the trie that are close to a given query string in terms of
    edit distance.
    
    See the paper "Tries for Approximate String Matching" by Shang and Merrett for details. This
    implementation assumes that we set an upper bound on the allowed edit distance (treating anything
    above this bound as infinity and non-retrievable), and that this upper bound is relatively small.
    Imposing a small upper bound allows us to prune the search space and make the search reasonably
    efficient.
    """

    @dataclass
    class Options:
        """
        Query-time options. Controls lookup behavior.
        """
        upper_bound: int = 1          # The maximum allowed edit distance between the query and a match.
        candidate_count: int = 10000  # The maximum number of candidate matches we score.
        hit_count: int = 10           # The maximum number of scored matches we will emit.
        first_n: int = 0              # Assume that the first N query characters are correct, to reduce the search space.
        scoring: str = "normalized"   # The scoring function to apply to candidate matches.

    @dataclass
    class Result:
        """
        An individual lookup result, as reported back to the client.
        """
        match: str        # The matching dictionary entry.
        meta: Any | None  # Optional meta data associated with the match, if present in the dictionary.
        score: float      # The score associated with the match, per the chosen scoring function.
        distance: int     # The edit distance between the query and the match.

    def __init__(self, trie: Trie, analyzer: Analyzer):
        self._trie = trie
        self._analyzer = analyzer  # The same as was used for trie building.

    def evaluate(self, query: str, options: Options | None = None) -> Iterator[Result]:
        """
        Locates all strings in the trie that are no more than a given number of edit errors away
        from the query string.

        The matching strings, if any, are scored and only the highest-scoring matches are yielded
        back to the client.
        """
        # Default options apply unless specified.
        options = options or self.Options()

        # Tokenize and join to be robust to nuances in whitespace.
        query = self._analyzer.join(query)

        # The upper bound for the edit distance we accept between the query and a match. Assumed to be
        # a small number, e.g., 1, 2, or 3. The lower we set the upper bound, the more we can prune
        # the search space, and the more efficient the lookup will be.
        upper_bound = max(0, options.upper_bound)

        # The maximum number of candidate matches we score.
        candidate_count = max(1, options.candidate_count)

        # The maximum number of scored matches we will emit.
        hit_count = max(1, min(100, options.hit_count))

        # Assume that the N first characters are correct? This significantly prunes down the search
        # space and can give a performance boost. However, we get worse recall if the assumption is
        # incorrect.
        first_n = max(0, min(len(query), options.first_n))

        # Make some modifications to our starting point, if needed.
        head = query[:first_n]
        tail = query[first_n:]
        root = self._trie if first_n == 0 else self._trie.consume(head)

        # The available scoring functions that the client can choose from. High
        # scores are better than low scores. The "lopresti" function is lifted
        # from https://www.cse.lehigh.edu/~lopresti/Publications/1996/sdair96.pdf.
        scorers = {
            "negated":    lambda d, q, c: -d,
            "normalized": lambda d, q, c: 1.0 - (d / (first_n + max(len(q), len(c)))),
            "lopresti":   lambda d, q, c: 1.0 / math.exp(d / (first_n + max(len(q), len(c)) - d)),
        }

        # The selected scoring function to apply to candidate matches.
        scorer = scorers.get(options.scoring, None)
        assert scorer is not None

        # For keeping track of scored candidate matches. Only retains the highest-scoring ones.
        sieve = Sieve(hit_count)

        # The edit table object that we update as we traverse the trie. Two strings that share
        # a prefix of length N also share the N first columns in the edit table. Hence, as we
        # traverse the trie we can avoid recomputing large parts of the table.
        table = EditTable(tail, "?" * 10, False)

        # Receives matches from the search, as they are found. The search aborts if the callback
        # returns False, i.e., when we have received sufficiently many candidate matches.
        # TODO: If the trie has meta data that indicates how common the entries are, use this for scoring.
        def callback(distance: int, candidate: str, meta: Any) -> bool:
            score = scorer(distance, tail, candidate)
            sieve.sift(score, (distance, candidate, meta))
            nonlocal candidate_count
            candidate_count -= 1
            return candidate_count > 0

        # Search! We receive and sift results via the callback.
        if root:
            self._dfs(root, 0, table, upper_bound, callback)

        # Emit the best matches!
        for score, (distance, match, meta) in sieve.winners():
            yield self.Result(head + match, meta, score, distance)

    def _dfs(self, node: Trie, level: int, table: EditTable, upper_bound: int, callback: Callable[[int, str, Any], bool]) -> bool:
        """
        Does a recursive depth-first search in the trie, pruning away paths that cannot lead
        to matches with a sufficiently low edit cost. See paper by Shang and Merrett for a
        detailed discussion.

        Returns True unless the supplied callback tells us to abort the search.

        As this implementation is recursive, the call stack might blow up if we go really
        many levels deep into the trie. That should not be an issue as the primary use case
        for this search is to consult a simple spellchecking dictionary of strings all having
        reasonable lengths, but could merit a second look if we look to apply this to other
        use cases.
        """
        # Are we at a node in the trie that corresponds to a dictionary entry?
        if node.is_final():

            # This may or may not be something that we want to report: We know that the
            # dictionary entry is within the edit distance bound to some prefix of the query
            # string, but the query string could be longer. So check the distance between the
            # dictionary entry and the complete query string. (We could easily do something
            # here to support different matching modes.)
            if (distance := table.distance(level)) <= upper_bound:
                if not callback(distance, table.prefix(level), node.get_meta()):
                    return False

        # The node may have children. Explore or prune the branches. Only explore if we have
        # not exceeded our upper bound. The lower our upper bound, the more of the search
        # space we can prune away.
        for transition in node.transitions(False):
            if table.update2(level + 1, transition) <= upper_bound:
                child = node.child(transition)
                if child and not self._dfs(child, level + 1, table, upper_bound, callback):
                    return False

        # Continue the search.
        return True


class WordEditSearchEngine:
    """
    Realizes a simple token-level edit distance lookup engine. For example, the edit distance
    between "foo bar baz" and "bar foo gog" is 2 with Damerau-Levenshtein distance, as at the
    token-level we can transpose the first two tokens and replace the third token.

    Internally, the searchable strings are transformed so that each unique token is mapped to
    a unique Unicode code point. We can then use a character-level edit distance engine over
    the transformed strings using a transformed query, and reverse the transformation when
    reporting results.
    """

    # The code point to use when a query token is encountered that is unknown, i.e., not present
    # in searchable data and thus not mappable to a known code point.
    _UNKNOWN = 0

    def __init__(self, strings: Iterable[str], analyzer: Analyzer):
        self._token2codepoint: Dict[str, int] = {}
        self._codepoint2token: Dict[int, str] = {}
        self._analyzer = analyzer
        self._engine = self._create_engine(strings)

    def _create_engine(self, strings: Iterable[str]) -> EditSearchEngine:
        """
        Creates the underlying character-level edit search engine, after transforming the
        supplied strings into sequences of code points.
        """
        dummy = DummyAnalyzer()
        trie = SimpleTrie.from_strings(list(self._transform(strings, self._analyzer)), dummy)
        return EditSearchEngine(trie, dummy)

    def _transform(self, strings: Iterable[str], analyzer: Analyzer) -> Iterator[str]:
        """
        Transforms each input string into a sequence of code points, one code point per token.
        The internal mappings between tokens and code points are updated as new tokens are
        encountered. We avoid the code point used to represent unknown query tokens.

        For convenience and debuggability we stick to using code points that map to printable
        symbols, but this is not strictly necessary.
        """
        codepoints = (i for i in range(0x110000) if chr(i).isprintable() and i != self._UNKNOWN)
        for string in strings:
            tokens = [token for token, _ in analyzer.terms(string)]
            for token in tokens:
                if token not in self._token2codepoint:
                    codepoint = next(codepoints)
                    self._token2codepoint[token] = codepoint
                    self._codepoint2token[codepoint] = token
            yield "".join(chr(self._token2codepoint[token]) for token in tokens)

    def _untransform(self, string: str) -> str:
        """
        Reverses the transformation of a code point sequence back into a token sequence.
        Assumes that all code points in the input string are known.
        """
        return " ".join(self._codepoint2token[ord(c)] for c in string)

    def evaluate(self, query: str, options: EditSearchEngine.Options | None = None) -> Iterator[EditSearchEngine.Result]:
        """
        Locates all strings that are no more than a given number of edit errors away from the
        query string. Edit distance is measured at the token-level.

        The matching strings, if any, are scored and only the highest-scoring matches are yielded
        back to the client.
        """
        # Map all query terms to code points. All unknown terms are mapped to a reserved code point,
        # we currently don't discern between different unknown terms.
        query = "".join(chr(self._token2codepoint.get(token, self._UNKNOWN)) for token, _ in self._analyzer.terms(query))

        # Do a character-level edit distance search. Reverse the transformation when reporting results.
        for result in self._engine.evaluate(query, options):
            yield EditSearchEngine.Result(self._untransform(result.match), result.meta, result.score, result.distance)
