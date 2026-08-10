# pylint: disable=missing-module-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=unnecessary-pass

from abc import ABC, abstractmethod
from typing import Iterator, List
from .posting import Posting
from .variablebytecodec import VariableByteCodec


class PostingList(ABC):
    """
    Abstract base class for a simple posting list.
    """

    def __iter__(self):
        return self.get_iterator()

    def __len__(self):
        return self.get_length()

    @abstractmethod
    def get_length(self) -> int:
        """
        Returns the length of the posting list, i.e., the number of postings it contains.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_iterator(self) -> Iterator[Posting]:
        """
        Returns an iterator that can be used to iterate over the posting list.
        """
        raise NotImplementedError()

    @abstractmethod
    def append_posting(self, posting: Posting) -> None:
        """
        Appends a posting to the posting list. It is up to the posting list implementation
        whether or not to append the posting immediately, or if the postings should be batched
        up internally into chunks and appended on a per chunk basis. E.g., for compressed
        posting lists some compression techniques might require or benefit from chunking.
        """
        raise NotImplementedError()

    @abstractmethod
    def finalize_postings(self) -> None:
        """
        Signals that there will be no more postings added to the posting list, e.g.,
        so that we can finalize any unprocessed chunks or otherwise tie up any loose
        ends.
        """
        raise NotImplementedError()


class InMemoryPostingList(PostingList):
    """
    A simple in-memory implementation of a posting list. Enforces that individual
    postings are sorted in ascending order by their document identifiers.

    A more realistic implementation would be as a contiguous buffer with binary and
    compressed posting data instead of as a list of objects. That would reduce memory
    fragmentation and memory use, and the buffer could easily be persisted and
    memory-mapped by the operating system.
    """

    def __init__(self) -> None:
        self._postings: List[Posting] = []

    def get_length(self) -> int:
        return len(self._postings)

    def get_iterator(self) -> Iterator[Posting]:
        return iter(self._postings)

    def append_posting(self, posting: Posting) -> None:
        assert len(self._postings) == 0 or self._postings[-1].document_id < posting.document_id
        self._postings.append(posting)

    def finalize_postings(self) -> None:
        pass


class CompressedInMemoryPostingList(PostingList):
    """
    A simple in-memory implementation of a compressed posting list. Combines simple gap encoding
    with variable-byte encoding.

    See Sections 5.3 and 5.3.1 in https://nlp.stanford.edu/IR-book/pdf/05comp.pdf for details.
    """

    class CompressedInMemoryPostingListIterator(Iterator[Posting]):
        """
        A custom iterator that decodes the compressed integers as we traverse the underlying byte
        array. The decoding logic needs to mirror the encoding logic that happens when postings are
        appended to the byte array.
        """

        def __init__(self, data: bytearray):
            self._data = data      # The buffer holding all the compressed posting data.
            self._where = 0        # Our current position in the buffer.
            self._document_id = 0  # We encoded the gaps, so accumulate them when decoding.

        def __next__(self) -> Posting:
            if self._where < len(self._data):
                gap, increment = VariableByteCodec.decode(self._data, self._where)
                self._where += increment
                self._document_id += gap
                term_frequency, increment = VariableByteCodec.decode(self._data, self._where)
                self._where += increment
                return Posting(self._document_id, term_frequency)
            raise StopIteration

    def __init__(self):
        self._logical_length = 0        # The number of posting entries encoded in the byte array.
        self._previous_document_id = 0  # So that we can gap encode.
        self._data = bytearray()        # All posting entries, compressed.

    def get_length(self) -> int:
        return self._logical_length

    def get_iterator(self) -> Iterator[Posting]:
        return self.CompressedInMemoryPostingListIterator(self._data)

    def append_posting(self, posting: Posting) -> None:
        assert self._logical_length == 0 or posting.document_id > self._previous_document_id
        gap = posting.document_id - self._previous_document_id
        VariableByteCodec.encode(gap, self._data)
        VariableByteCodec.encode(posting.term_frequency, self._data)
        self._logical_length += 1
        self._previous_document_id = posting.document_id

    def finalize_postings(self) -> None:
        pass
