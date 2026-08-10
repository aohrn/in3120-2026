# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=protected-access

from __future__ import annotations
import mmap
from typing import Any, Iterator
from .trie import Trie
from .triepacker import TriePacker


class PackedTrie(Trie):
    """
    A packed trie that resides on disk and that we can memory-map. For background details, see the
    paper "Tightly Packed Tries: How to Fit Large Models into Memory, and Make them Load Fast, Too"
    by Germann, Joanis, and Larkin.

    This class implements the context management protocol: The root should ideally be used as part
    of a 'with' statement, so that the underlying resource gets properly closed.
    """

    def __init__(self, filename: str | None):
        self._view: mmap.mmap | None = None
        self._data: TriePacker.PackedNode | None = None
        self._is_root: bool = filename is not None
        if self._is_root:
            with open(filename or "", "r+b") as file:
                self._view = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            root, _ = TriePacker.read_header(self._view)
            self._data, _ = TriePacker.read_node(self._view, root)

    def __enter__(self) -> PackedTrie:
        if self._view is None:
            raise RuntimeError("No mapped memory.")
        return self

    def __exit__(self, *args) -> None:
        self._close()

    def __del__(self) -> None:
        self._close()

    def _close(self) -> None:
        """
        Clean-up. Closes underlying resources when we dispose of the root node.
        """
        if self._is_root and self._view is not None:
            self._view.close()
            self._view = None
        self._data = None

    def _advance(self, offset: int) -> PackedTrie:
        """
        Returns the node reached by reading the data at the given offset.
        """
        assert self._view is not None
        node = PackedTrie(None)
        node._view = self._view
        node._data, _ = TriePacker.read_node(self._view, offset)
        return node

    def child(self, transition: str) -> Trie | None:
        assert self._data is not None
        return next((self._advance(offset) for symbol, offset in self._data.children if transition == symbol), None)

    def transitions(self, sort: bool = True) -> Iterator[str]:
        assert self._data is not None
        yield from (s for s, _ in self._data.children)  # Already sorted.

    def is_final(self) -> bool:
        assert self._data is not None
        return self._data.is_final

    def get_meta(self) -> Any | None:
        assert self._data is not None
        return self._data.meta
