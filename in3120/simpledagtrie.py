# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=protected-access

from __future__ import annotations
from typing import Any, Iterable, Tuple, List
from itertools import repeat
from .analyzer import Analyzer
from .trie import SimpleTrie


class SimpleDAGTrie(SimpleTrie):
    """
    A trie that internally also shares suffixes in addition to prefixes, effectively implementing a directed
    acyclic graph (DAG) structure. In automata theory, it'd be called a deterministic acyclic finite-state
    automaton.
    
    For inspiration, see, e.g., the paper "How to Squeeze a Lexicon" by Ciura and Deorowicz. See also
    https://en.wikipedia.org/wiki/Deterministic_acyclic_finite_state_automaton for background.

    A further space-preserving optimization would be to compress paths (wherever they might occur in the trie) such
    that chains of nodes that only have a single child are merged into a single node. This is not implemented here.
    Instead of single-symbol transitions between nodes we could then end up with longer transition strings between
    nodes, and would need to adjust the traversal logic accordingly. See, e.g., https://en.wikipedia.org/wiki/Radix_tree
    for details.
    """

    def __init__(self):
        super().__init__()
        self._register = {}    # Maps node signatures to nodes.
        self._previous = ""    # The previously added string.
        
    @staticmethod
    def from_strings(strings: Iterable[str], analyzer: Analyzer) -> SimpleDAGTrie:
        """
        Constructor-like convenience method. Creates and returns a new trie containing
        all the given strings.
        """
        return SimpleDAGTrie.from_strings2(zip(strings, repeat(None)), analyzer)

    @staticmethod
    def from_strings2(strings: Iterable[Tuple[str, None | Any]], analyzer: Analyzer) -> SimpleDAGTrie:
        """
        Constructor-like convenience method. Creates and returns a new trie containing
        all the given (string, meta) pairs.
        """
        root = SimpleDAGTrie()
        root.add2(strings, analyzer)
        return root

    def _add(self, string: str, meta: None | Any) -> None:
        """
        Overridden.
        """
        assert self._previous < string, "Strings are assumed added in lexicographic order."
        assert meta is None, "Support for meta data is not implemented."
        common = next((i for i, (a, b) in enumerate(zip(string, self._previous)) if a != b), min(len(string), len(self._previous)))
        if self._previous: self._minimize(common)
        super()._add(string, meta)
        self._previous = string

    def add2(self, strings: Iterable[Tuple[str, None | Any]], analyzer: Analyzer) -> None:
        """
        Overridden. Ensures that the strings are added in the right order, and triggers minimization
        for the string that was added last.
        """
        assert len(self._children) == 0, "Only addition of strings to an empty trie is supported."
        super().add2(sorted(strings), analyzer)
        if self._previous:
            self._minimize(0)
            self._previous = ""
            self._register.clear()

    def _minimize(self, common: int) -> None:
        """
        Minimize nodes along the suffix of the previously added string, given that we know the length
        of the shared prefix between the previously added string and the currently added string.

        Performs suffix sharing by replacing children with previously added nodes that are equivalent
        and safe to share. For example, if we have the two strings "bilerall" and "båterall", not only
        can they share the prefix "b" but they can also share the suffix "erall".
        """
        # For the previously inserted word, build the path from the root to the leaf.
        stack = []
        node = self
        for symbol in self._previous:
            stack.append((node, symbol))
            node = node[symbol]

        # Start at the leaf and work backwards, but stop when we reach the common prefix.
        # If we have the opportunity to reuse nodes, do so.
        while len(stack) > common:
            parent, symbol = stack.pop()
            child = parent.child(symbol)
            signature = self._signature(child)
            if signature in self._register:
                parent._children[symbol] = self._register[signature]
            else:
                self._register[signature] = child

    def _signature(self, node: SimpleTrie) -> int:
        """
        Computes a canonical signature for a node.
        
        Uses the built-in hash function to generate the signature. We assume that the chance of
        collisions is low enough to be acceptable in practice.
        
        We're using the id function for convenience, rather than recursively considering the signature
        of the children. This wouldn't work for all trie implementations.
        """
        items: List[Any] = [node.is_final()]
        for child, symbol in node.children(True):
            items.append((symbol, id(child)))
        return hash(tuple(items))
