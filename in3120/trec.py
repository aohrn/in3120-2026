# pylint: disable=missing-module-docstring

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set
from .corpus import Corpus, InMemoryCorpus
from .document import Document
from .corpusloader import CorpusLoader


class TREC:
    """
    Represents a set of documents, a set of queries, and a set of relevance
    judgments. Supports loading these from files in TREC format.

    See Section 8.2 in https://nlp.stanford.edu/IR-book/pdf/08eval.pdf and
    https://github.com/oussbenk/cranfield-trec-dataset/blob/main/README.md
    for details. See also https://en.wikipedia.org/wiki/Text_Retrieval_Conference
    for more information about TREC in general.
    """

    # Maps from TREC topic/query identifiers to sets of TREC document identifiers,
    # indicating which documents are relevant to which queries. Note that
    # TREC document identifiers are not necessarily the same as the document
    # identifiers in the document corpus, and similarly for queries.
    Judgments = Dict[str, Set[str]]

    @dataclass
    class Options:
        """
        Configuration options for TREC datasets.
        """
        document_id_field: str = "docno"  # The named field that holds the TREC document identifier.
        query_id_field: str = "num"       # The named field that holds the TREC topic/query identifier.

    def __init__(self, documents: str, queries: str, judgments: str, options: Options | None = None):
        self._options = options or self.Options()
        self.documents: Corpus = CorpusLoader.from_files(InMemoryCorpus(), [documents])
        self.queries: Corpus = CorpusLoader.from_files(InMemoryCorpus(), [queries])
        self.judgments: TREC.Judgments = self._load_judgments(judgments)

    def _load_judgments(self, filename: str) -> TREC.Judgments:
        """
        Loads the relevance judgments from the given file. The judgments are returned as a
        mapping from TREC topic/query identifiers to sets of TREC document identifiers that
        are relevant to that TREC topic/query.
        """
        judgments: TREC.Judgments = {}
        with open(filename) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                topic, _, docno, relevance = parts
                if relevance == "1":
                    judgments.setdefault(topic, set()).add(docno)
        return judgments

    def is_relevant(self, query: Document, document: Document) -> bool:
        """
        Returns True iff the given document is relevant to the given query, according to the
        (binarized) TREC relevance judgments. Both the query and the document are assumed to
        be members of the TREC dataset.
        """
        query_id = query.get_field(self._options.query_id_field, "")
        document_id = document.get_field(self._options.document_id_field, "")
        return document_id in self.judgments.get(query_id, set())

    def relevant_count(self, query: Document) -> int:
        """
        Returns the number of relevance judgments for the given TREC query. The query is assumed
        to be a member of the TREC dataset.
        """
        query_id = query.get_field(self._options.query_id_field, "")
        return len(self.judgments.get(query_id, set()))
