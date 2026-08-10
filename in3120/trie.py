# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=protected-access
# pyright: reportOptionalMemberAccess=false

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import repeat
from typing import Callable, Dict, Any, Set, Tuple, List, Iterable, Iterator
from .analyzer import Analyzer


class Trie(ABC):
    """
    Simple abstract base class for character-level tries. A node in the trie is also itself a trie.

    We can, optionally, associate a meta data value of some kind with each final/terminal state in
    the trie/automaton. Some applications might benefit from this, if we need to keep rich meta data
    associated with each string. In such cases, these associated values could be other strings, or
    values that can be used as lookup keys into, e.g., some external database. If a string to add
    has no meta data associated with it, we associate the value None.
    """

    @dataclass
    class Counts:
        """
        Descriptive statistics. For debugging and testing purposes, mostly.
        """
        node: int     # The number of nodes in the trie.
        unique: int   # The number of unique nodes, according to a client-supplied function.
        final: int    # The number of logical final/terminal states in the trie.
        meta: int     # The number of logical final/terminal states that have meta data associated with them.

    def __contains__(self, string: str) -> bool:
        descendant = self.consume(string)
        return descendant is not None and descendant.is_final()

    def __iter__(self) -> Iterator[str]:
        return self.strings()

    def __getitem__(self, prefix: str) -> Trie:
        node = self.consume(prefix)
        if node is None:
            raise KeyError(f"Prefix '{prefix}' not found.")
        return node

    @abstractmethod
    def child(self, transition: str) -> Trie | None:
        """
        Returns the immediate child node, given a transition symbol. Returns None if the transition
        symbol is invalid. Functionally equivalent to consume(transition), but simpler and for the
        special case of a single transition symbol and not a longer string.

        Assumes that the transition symbol is already normalized.
        """
        raise NotImplementedError()

    @abstractmethod
    def transitions(self, sort: bool = True) -> Iterator[str]:
        """
        Returns the set of symbols that are valid outgoing transitions, i.e., the set of symbols that
        when consumed by this node would lead to a valid child node. The returned transitions are
        emitted back in lexicographical order, if specified.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_final(self) -> bool:
        """
        Returns True iff the current node is a final/terminal state in the trie/automaton, i.e.,
        if a string has been added to the trie where the end of the string ends up in this node.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_meta(self) -> Any | None:
        """
        Returns the meta data associated with the final/terminal state, or None if no such meta
        data exists.
        """
        raise NotImplementedError()

    def consume(self, prefix: str) -> Trie | None:
        """
        Consumes the given prefix verbatim and returns the resulting descendant node,
        if any. I.e., if strings that have this prefix have been added to the trie, then
        the trie node corresponding to traversing the prefix is returned. Otherwise, None
        is returned.

        Assumes that the prefix is already normalized.
        """
        node = self
        for symbol in prefix:
            node = node.child(symbol)
            if node is None:
                return None
        return node

    def children(self, sort: bool = False) -> Iterator[Tuple[Trie, str]]:
        """
        Yields all immediate children and their associated transition symbols.
        The returned children are emitted back in lexicographical order by their
        associated transition symbols, if specified.
        """
        for symbol in self.transitions(sort):
            yield self.child(symbol), symbol # pyright: ignore[reportReturnType]

    def strings(self) -> Iterator[str]:
        """
        Yields all strings that are found in or below this node. For simple testing and debugging purposes.
        The returned strings are emitted back in lexicographical order.
        """
        stack = [(self, "")]
        while stack:
            node, prefix = stack.pop()
            if node.is_final():
                yield prefix
            for symbol in reversed(list(node.transitions(True))):
                stack.append((node.child(symbol), prefix + symbol)) # pyright: ignore[reportArgumentType]

    def counts(self, signature: Callable[[Trie], int] = id) -> Counts:
        """
        Computes and returns some descriptive statistics about the trie rooted at this node.
        Node counts include the root node.

        A custom signature function can be supplied that maps each trie node to an integer, and
        the number of unique values generated by this signature function is returned. The default
        signature function is the built-in id function.
        """
        counts = self.Counts(0, 0, 0, 0)
        seen = set()
        stack: List[Trie] = [self]
        while stack:
            node = stack.pop()
            counts.node += 1
            seen.add(signature(node))
            if node.is_final():
                counts.final += 1
                if node.has_meta():
                    counts.meta += 1
            stack.extend(child for child, _ in node.children(False))
        counts.unique = len(seen)
        return counts

    def to_dot(self, signature: Callable[[Trie], int] = id) -> str:
        """
        Returns a string representation of the trie in DOT format, suitable for visualization
        and debugging. For online tools that can render DOT see, e.g.:

        - https://dreampuf.github.io/GraphvizOnline/
        - https://edotor.net/
        - https://www.webgraphviz.com/
        - https://www.tools-online.app/tools/graphviz

        This implementation also works for subclasses that do suffix sharing in addition to
        prefix sharing, as long as a suitable signature function is supplied that maps unique
        nodes to unique integers. The default signature function is the built-in id function.
        """
        lines: List[str] = ["digraph {"]

        # All unique nodes.
        stack: List[Trie] = [self]
        seen: Set[int] = set()
        while stack:
            node = stack.pop()
            name = signature(node)
            if name in seen:
                continue
            seen.add(name)
            if node.is_final():
                if node.has_meta():
                    lines.append(f'  "{name}" [shape=doublecircle, label="{node.get_meta()}"];')
                else:
                    lines.append(f'  "{name}" [shape=doublecircle, label=""];')
            else:
                lines.append(f'  "{name}" [shape=circle, label=""];')
            stack.extend(child for child, _ in node.children(False))

        # All unique edges.
        stack = [self]
        seen = set()
        while stack:
            node = stack.pop()
            name = signature(node)
            if name in seen:
                continue
            seen.add(name)
            for child, symbol in node.children(False):
                lines.append(f'  "{name}" -> "{signature(child)}" [label="{symbol}"];')
                stack.append(child)

        lines.append("}")
        return "\n".join(lines)

    def has_meta(self) -> bool:
        """
        Returns True iff the current node is a final/terminal state that has meta data associated
        with it.
        """
        return self.get_meta() is not None


class SimpleTrie(Trie):
    """
    A very simple and straightforward implementation of a trie for demonstration purposes
    and tiny dictionaries.

    A serious real-world implementation of a trie or an automaton would not be implemented
    this way. The trie/automaton would then instead be encoded into a single contiguous buffer
    and there'd be significant attention on memory consumption and scalability with respect to
    dictionary size.

    Some plausible open source alternatives include, e.g.:

    - Marisa (https://github.com/pytries/marisa-trie)
    - DAWG (https://dawg.readthedocs.io/en/latest/)
    - datrie (https://pypi.org/project/datrie/)
    - hat-trie (https://github.com/pytries/hat-trie)
    - dafsa (https://github.com/tresoldi/dafsa
    """

    # Reduce the memory footprint of objects by preventing the creation of a dynamic dictionary
    # for instance attributes.
    __slots__ = ["_children"]

    def __init__(self):
        self._children: Dict[str, None | SimpleTrie] = {}

    @staticmethod
    def from_strings(strings: Iterable[str], analyzer: Analyzer) -> SimpleTrie:
        """
        Constructor-like convenience method. Creates and returns a new trie containing
        all the given strings.
        """
        return SimpleTrie.from_strings2(zip(strings, repeat(None)), analyzer)

    @staticmethod
    def from_strings2(strings: Iterable[Tuple[str, None | Any]], analyzer: Analyzer) -> SimpleTrie:
        """
        Constructor-like convenience method. Creates and returns a new trie containing
        all the given (string, meta) pairs.
        """
        root = SimpleTrie()
        root.add2(strings, analyzer)
        return root

    def _add(self, string: str, meta: None | Any) -> None:
        """
        Internal helper method, adds the given non-empty string and its optional
        associated meta data to the trie with this node as the root. The string is
        assumed already properly normalized at this point.

        The special transition symbol "" is used as a marker to indicate that a node
        is final/terminal. The meta data, if any, is associated with this special
        transition symbol.
        """
        trie = self
        for symbol in string:
            if symbol not in trie._children:
                trie._children[symbol] = SimpleTrie()
            trie = trie._children[symbol]  # type: ignore[assignment]
        if "" in trie._children:
            assert trie._children[""] == meta
        else:
            trie._children[""] = meta

    def add(self, strings: Iterable[str], analyzer: Analyzer) -> None:
        """
        Adds all the strings to the trie, after normalizing them. The tokenizer is used so
        that we're robust to nuances in whitespace and punctuation.

        Adding the same string more than once is benign and idempotent. Note that "same" here
        means after normalization.
        """
        self.add2(zip(strings, repeat(None)), analyzer)

    def add2(self, strings: Iterable[Tuple[str, None | Any]], analyzer: Analyzer) -> None:
        """
        Adds all the strings and their associated meta data values to the trie,
        after normalizing them. The tokenizer is used so that we're robust to nuances
        in whitespace and punctuation.

        If a string has no meta data associated with it, None is assumed passed as the
        meta data value.

        Adding the same string more than once is benign and idempotent, as long as their
        associated meta data values do not differ. Note that "same" here means after
        normalization.
        """
        for string, meta in strings:
            assert string is not None
            self._add(analyzer.join(string), meta)

    def child(self, transition: str) -> None | Trie:
        return self._children.get(transition, None)

    def transitions(self, sort: bool = True) -> Iterator[str]:
        children = (s for s in self._children if s)
        yield from sorted(children) if sort else children

    def is_final(self) -> bool:
        return "" in self._children

    def get_meta(self) -> None | Any:
        return self._children[""] if self.is_final() else None
