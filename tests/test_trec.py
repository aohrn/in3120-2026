# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestTREC(unittest.TestCase):

    def setUp(self):
        documents = "../data/trec/cranfield/cran.all.1400.xml"
        queries   = "../data/trec/cranfield/cran.qry.xml"
        judgments = "../data/trec/cranfield/cranqrel.trec.txt"
        self._cranfield = in3120.TREC(documents, queries, judgments)

    def test_overall_structure_cranfield(self):
        self.assertIsNotNone(self._cranfield.documents)
        self.assertEqual(self._cranfield.documents.size(), 1400)
        self.assertSetEqual(set(self._cranfield.documents[0].get_field_names()), set(["docno", "title", "author", "bib", "text"]))
        self.assertIsNotNone(self._cranfield.queries)
        self.assertEqual(self._cranfield.queries.size(), 225)
        self.assertSetEqual(set(self._cranfield.queries[0].get_field_names()), set(["num", "title"]))
        self.assertIsNotNone(len(self._cranfield.judgments), 225)

    def test_document_structure_cranfield(self):
        document = self._cranfield.documents[0]
        self.assertEqual(document.get_field("docno", ""), "1")
        self.assertEqual(document.get_field("title", ""), "experimental investigation of the aerodynamics of a\nwing in a slipstream .")
        self.assertEqual(document.get_field("author", ""), "brenckman,m.")
        self.assertEqual(document.get_field("bib", ""), "j. ae. scs. 25, 1958, 324.")
        self.assertTrue(document.get_field("text", "").endswith("destalling effects was made for\nthe specific configuration of the experiment ."))

    def test_query_structure_cranfield(self):
        query = self._cranfield.queries[0]
        self.assertEqual(query.get_field("num", ""), "1")
        self.assertEqual(query.get_field("title", ""), "what similarity laws must be obeyed when constructing aeroelastic models\nof heated high speed aircraft .")

    def test_judgment_structure_cranfield(self):
        query = self._cranfield.queries[0]  # TREC query identifier "1".
        relevant = self._cranfield.judgments["1"]
        self.assertEqual(len(relevant), 28)
        self.assertEqual(self._cranfield.relevant_count(query), 28)
        self.assertSetEqual(relevant, set(["184", "29", "31", "12", "51", "102", "13", "14", "15", "57", "378", "859", "185", "30", "37", "52", "142", "195", "875", "56", "66", "95", "462", "497", "858", "876", "879", "880"]))

    def test_is_relevant_cranfield(self):
        query = self._cranfield.queries[0]
        relevant = [d for d in self._cranfield.documents if d["docno"] in self._cranfield.judgments[query.get_field("num", "")]]
        irrelevant = [d for d in self._cranfield.documents if d["docno"] not in self._cranfield.judgments[query.get_field("num", "")]]
        for document in relevant:
            self.assertTrue(self._cranfield.is_relevant(query, document))
        for document in irrelevant:
            self.assertFalse(self._cranfield.is_relevant(query, document))


if __name__ == '__main__':
    unittest.main(verbosity=2)
