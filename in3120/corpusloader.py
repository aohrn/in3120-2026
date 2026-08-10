# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from __future__ import annotations
import csv
from json import loads
from typing import Any, Dict, List
from xml.etree import ElementTree as ET
from .corpus import Corpus
from .document import InMemoryDocument
from .documentpipeline import DocumentPipeline


class CorpusLoader:
    """
    Helper class for loading a document collection.
    """

    @staticmethod
    def from_files(corpus: Corpus, filenames: List[str],
                   annotations: Dict[str, Any] | List[Dict[str, Any]] | None = None,
                   pipeline: DocumentPipeline | None = None) -> Corpus:
        """
        Loads a set of documents from a set of files, and adds the documents to the given corpus.
        The populated corpus is returned, for convenience. Assigns document identifiers on a first-come
        first-serve basis.

        For each filename supplied, if any, the client can also supply a dictionary of "annotations"
        that each document from that file gets assigned as additional content. This is useful so that
        it becomes easy to identify the origin of each document, e.g., when splitting the corpus into
        multiple corpora that define a training set for a classifier.

        Optionally, the client can supply a document processing pipeline that is applied to every
        document. The processing pipeline can transform or even drop a document before it gets
        added to the corpus.
        """
        assert corpus is not None
        assert filenames is not None
        if isinstance(annotations, dict):
            annotations = [annotations for _ in range(len(filenames))]
        if annotations is None:
            annotations = [{} for _ in range(len(filenames))]
        assert len(filenames) == len(annotations)
        pipeline = pipeline or DocumentPipeline([])
        for filename, annotation in zip(filenames, annotations):
            assert filename is not None
            assert annotation is not None
            if filename.endswith(".txt"):
                CorpusLoader._from_text_file(corpus, filename, annotation, pipeline)
            elif filename.endswith(".xml"):
                CorpusLoader._from_xml_file(corpus, filename, annotation, pipeline)
            elif filename.endswith(".json"):
                CorpusLoader._from_json_file(corpus, filename, annotation, pipeline)
            elif filename.endswith(".csv"):
                CorpusLoader._from_csv_or_tsv_file(corpus, filename, ",", annotation, pipeline)
            elif filename.endswith(".tsv"):
                CorpusLoader._from_csv_or_tsv_file(corpus, filename, "\t", annotation, pipeline)
            else:
                raise IOError(f"Filename has unsupported extension: {filename}")
        return corpus

    @staticmethod
    def _from_text_file(corpus: Corpus, filename: str, annotation: Dict[str, Any], pipeline: DocumentPipeline) -> None:
        """
        Loads documents from the given UTF-8 encoded text file. One document per line,
        tab-separated fields. Empty lines are ignored. The first field gets named "body",
        the second field (optional) gets named "meta". All other fields are currently ignored.
        """
        document_id = corpus.size()
        with open(filename, mode="r", encoding="utf-8") as file:
            for line in file:
                anonymous_fields = line.strip().split("\t")
                if len(anonymous_fields) == 1 and not anonymous_fields[0]:
                    continue
                named_fields = {"body": anonymous_fields[0]}
                if len(anonymous_fields) >= 2:
                    named_fields["meta"] = anonymous_fields[1]
                named_fields.update(annotation)
                document = pipeline(InMemoryDocument(document_id, named_fields))
                if document:
                    corpus.add_document(document)
                    document_id += 1

    @staticmethod
    def _from_xml_file(corpus: Corpus, filename: str, annotation: Dict[str, Any], pipeline: DocumentPipeline) -> None:
        """
        Loads documents from the given XML file. The XML file is expected to have a single root element,
        with one document per tag contained under the root element.
        """
        def _get_text(element: ET.Element) -> str:
            return " ".join(t.strip() for t in element.itertext() if t.strip())
        tree = ET.parse(filename)
        root = tree.getroot()
        document_id = corpus.size()
        for record in root:
            if not isinstance(record.tag, str):
                continue
            named_fields = {}
            for field in record:
                if not isinstance(field.tag, str):
                    continue
                named_fields[field.tag] = _get_text(field)
            if not named_fields:
                named_fields = {"body": _get_text(record)}
            named_fields.update(annotation)
            document = pipeline(InMemoryDocument(document_id, named_fields))
            if document:
                corpus.add_document(document)
                document_id += 1

    @staticmethod
    def _from_csv_or_tsv_file(corpus: Corpus, filename: str, delimiter: str, annotation: Dict[str, Any], pipeline: DocumentPipeline) -> None:
        """
        Loads documents from the given UTF-8 encoded CSV file. One document per line.
        """
        document_id = corpus.size()
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                named_fields = dict(row)
                named_fields.update(annotation)
                document = pipeline(InMemoryDocument(document_id, named_fields))
                if document:
                    corpus.add_document(document)
                    document_id += 1

    @staticmethod
    def _from_json_file(corpus: Corpus, filename: str, annotation: Dict[str, Any], pipeline: DocumentPipeline) -> None:
        """
        Loads documents from the given UTF-8 encoded JSON file. One document per line.
        Lines that do not start with "{" and end with "}" are ignored.
        """
        document_id = corpus.size()
        with open(filename, mode="r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    named_fields = loads(line)
                    named_fields.update(annotation)
                    document = pipeline(InMemoryDocument(document_id, named_fields))
                    if document:
                        corpus.add_document(document)
                        document_id += 1
