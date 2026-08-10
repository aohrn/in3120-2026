# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
from context import in3120


class TestFeatureSelector(unittest.TestCase):

    def setUp(self):
        analyzer = in3120.SimpleAnalyzer()
        movies = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/imdb.csv"])
        fields = ["title", "description"]
        training_set = movies.split("genre", lambda v: v.split(","))
        stopwords0 = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/stopwords-en.txt"])
        self._stopwords = in3120.SimpleTrie.from_strings((d["body"] for d in stopwords0), analyzer)
        self._selector = in3120.FeatureSelector(training_set, fields, analyzer)

    def test_barfs_on_illegal_inputs(self):
        with self.assertRaises(AssertionError):
            list(self._selector.mutual_information(["WeirdCategory"], 10, self._stopwords))
        with self.assertRaises(AssertionError):
            list(self._selector.mutual_information(["Comedy"], -6, self._stopwords))
        with self.assertRaises(AssertionError):
            list(self._selector.chi_square(["BizarreCategory"], 10, self._stopwords))
        with self.assertRaises(AssertionError):
            list(self._selector.chi_square(["Comedy"], -8, self._stopwords))

    def test_mutual_information_on_movie_corpus(self):
        results = list(self._selector.mutual_information(["Horror", "Comedy", "Sci-Fi"], 10, self._stopwords))
        self.assertEqual(30, len(results))
        horror = [(r.term, r.score) for r in results if r.category == "Horror"]
        comedy = [(r.term, r.score) for r in results if r.category == "Comedy"]
        scifi  = [(r.term, r.score) for r in results if r.category == "Sci-Fi"]
        for pairs in (horror, comedy, scifi):
            self.assertEqual(10, len(pairs))
            self.assertTrue(all(pair[1] <= 1.0 for pair in pairs))
            self.assertTrue(pairs[i][1] >= pairs[i + 1][1] for i in range(len(pairs) - 1))
        self.assertListEqual([t for t, _ in horror], ["zombie", "cabin", "presence", "evil", "mysterious", "woods", "survivors", "victim", "remote", "unleashes"])
        self.assertListEqual([t for t, _ in comedy], ["party", "guy", "school", "friends", "kid", "throw", "responsibilities", "project", "fraternity", "dirty"])
        self.assertListEqual([t for t, _ in scifi],  ["alien", "earth", "future", "transformers", "tony", "stark", "iron", "intergalactic", "autobots", "aliens"])

    def test_chi_square_on_movie_corpus(self):
        results = list(self._selector.chi_square(["Horror", "Comedy", "Sci-Fi"], 10, self._stopwords))
        self.assertEqual(30, len(results))
        horror = [(r.term, r.score) for r in results if r.category == "Horror"]
        comedy = [(r.term, r.score) for r in results if r.category == "Comedy"]
        scifi  = [(r.term, r.score) for r in results if r.category == "Sci-Fi"]
        for pairs in (horror, comedy, scifi):
            self.assertEqual(10, len(pairs))
            self.assertTrue(pairs[i][1] >= pairs[i + 1][1] for i in range(len(pairs) - 1))
        self.assertListEqual([t for t, _ in horror], ["zombie", "cabin", "presence", "unleashes", "jennifer", "victim", "remote", "tourists", "stitching", "sequence"])
        self.assertListEqual([t for t, _ in comedy], ["party", "guy", "throw", "responsibilities", "project", "fraternity", "dirty", "school", "kid", "newly"])
        self.assertListEqual([t for t, _ in scifi],  ["alien", "earth", "transformers", "tony", "stark", "iron", "intergalactic", "autobots", "future", "2307"])

if __name__ == '__main__':
    unittest.main(verbosity=2)
