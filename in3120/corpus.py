# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from __future__ import annotations
import collections.abc
from abc import abstractmethod
from typing import Any, List, Dict, Callable, Set
from .document import Document


class Corpus(collections.abc.Iterable):
    """
    Abstract base class representing a corpus we can index and search over,
    i.e., a collection of documents. The class facilitates iterating over
    all documents in the corpus.
    """

    def __len__(self):
        return self.size()

    def __getitem__(self, document_id: int) -> Document:
        return self.get_document(document_id)

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def size(self) -> int:
        """
        Returns the number of documents in the document collection.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_document(self, document_id: int) -> Document:
        """
        Returns the document associated with the given document identifier.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_document(self, document: Document, strict: bool = True) -> None:
        """
        Adds the given document to the corpus.
        """
        raise NotImplementedError()


class InMemoryCorpus(Corpus):
    """
    An in-memory implementation of a document store, suitable only for small
    document collections.
    """

    def __init__(self):
        self._documents: List[Document] = []

    def __iter__(self):
        return iter(self._documents)

    def size(self) -> int:
        return len(self._documents)

    def get_document(self, document_id: int) -> Document:
        assert 0 <= document_id < len(self._documents)
        return self._documents[document_id]

    def add_document(self, document: Document, strict: bool = True) -> None:
        assert document is not None
        assert (not strict) or (document.document_id == len(self._documents))
        self._documents.append(document)

    def split(self, field_name: str, splitter: Callable[[Any], List[Any]] | None = None) -> Dict[Any, InMemoryCorpus]:
        """
        Divides the corpus up into multiple corpora, according to the value(s) of the
        named field. I.e., splits the corpus up into several smaller corpora.

        The value(s) of the named fields are used as keys for the splits. A custom splitter
        function can optionally be provided, in case the named field is multi-valued and/or the
        value(s) should be filtered or transformed in some way.
        """
        splitter = splitter if splitter else lambda v: [v]
        splits = {}
        for document in self:
            values = splitter(document.get_field(field_name, ""))
            for value in values:
                if value not in splits:
                    splits[value] = InMemoryCorpus()
                splits[value].add_document(document, False)
        return splits

    @staticmethod
    def merge(splits: Dict[Any, Corpus]) -> InMemoryCorpus:
        """
        The inverse of the split method. Merges a bunch of corpora together, deduplicating any shared
        documents.
        """
        unique = set()
        merged = InMemoryCorpus()
        for _, corpus in splits.items():
            for document in corpus:
                if document.document_id not in unique:
                    unique.add(document.document_id)
                    merged.add_document(document, False)
        return merged


class AccessLoggedCorpus(Corpus):
    """
    Wraps another corpus, and keeps an in-memory log of which documents
    that have been accessed. Facilitates testing.
    """

    def __init__(self, wrapped: Corpus):
        self._wrapped = wrapped
        self._accesses: Set[int] = set()

    def __iter__(self):
        return iter(self._wrapped)

    def size(self) -> int:
        return self._wrapped.size()

    def get_document(self, document_id: int) -> Document:
        self._accesses.add(document_id)
        return self._wrapped.get_document(document_id)

    def add_document(self, document: Document, strict: bool = True) -> None:
        return self._wrapped.add_document(document, strict)

    def get_history(self) -> Set[int]:
        """
        Returns the set of document identifiers that clients have accessed so far.
        """
        return self._accesses
