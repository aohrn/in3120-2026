# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestInMemoryCorpus(unittest.TestCase):

    def test_access_documents(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"body": "this is a Test"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"title": "prØve", "body": "en to tre"}))
        self.assertEqual(corpus.size(), 2)
        self.assertListEqual([d.document_id for d in corpus], [0, 1])
        self.assertListEqual([corpus[i].document_id for i in range(0, corpus.size())], [0, 1])
        self.assertListEqual([corpus.get_document(i).document_id for i in range(0, corpus.size())], [0, 1])

    def test_split_without_splitter(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"category": "A", "body": "Document zero"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"category": "A", "body": "Document one"}))
        corpus.add_document(in3120.InMemoryDocument(2, {"category": "B", "body": "Document two"}))
        corpus.add_document(in3120.InMemoryDocument(3, {"category": "C", "body": "Document three"}))
        splits = corpus.split("category")
        self.assertListEqual(["A", "B", "C"], sorted(splits.keys()))
        self.assertListEqual([d.document_id for d in splits["A"]], [0, 1])
        self.assertListEqual([d.document_id for d in splits["B"]], [2])
        self.assertListEqual([d.document_id for d in splits["C"]], [3])

    def test_split_with_splitter_and_then_merge(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"category": "X,Y", "body": "Document zero"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"category": "X", "body": "Document one"}))
        corpus.add_document(in3120.InMemoryDocument(2, {"category": "Z", "body": "Document two"}))
        corpus.add_document(in3120.InMemoryDocument(3, {"category": "Y", "body": "Document three"}))
        splits = corpus.split("category", lambda v: v.split(","))
        self.assertListEqual(["X", "Y", "Z"], sorted(splits.keys()))
        self.assertListEqual([d.document_id for d in splits["X"]], [0, 1])
        self.assertListEqual([d.document_id for d in splits["Y"]], [0, 3])
        self.assertListEqual([d.document_id for d in splits["Z"]], [2])
        merged = in3120.InMemoryCorpus.merge(splits)
        self.assertEqual(merged.size(), 4)
        self.assertListEqual(sorted([d.document_id for d in merged]), [0, 1, 2, 3])

    def test_split_with_splitter_that_transforms(self):
        corpus = in3120.InMemoryCorpus()
        corpus.add_document(in3120.InMemoryDocument(0, {"category": "X", "body": "Document zero"}))
        corpus.add_document(in3120.InMemoryDocument(1, {"category": "X", "body": "Document one"}))
        corpus.add_document(in3120.InMemoryDocument(2, {"category": "Z", "body": "Document two"}))
        corpus.add_document(in3120.InMemoryDocument(3, {"category": "Y", "body": "Document three"}))
        splits = corpus.split("category", lambda v: ["match"] if "X" in v else ["not match"])
        self.assertListEqual(["match", "not match"], sorted(splits.keys()))
        self.assertListEqual([d.document_id for d in splits["match"]], [0, 1])
        self.assertListEqual([d.document_id for d in splits["not match"]], [2, 3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
