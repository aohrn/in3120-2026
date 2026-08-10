# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestNearestNeighborClassifier(unittest.TestCase):

    def setUp(self):
        analyzer = in3120.SimpleAnalyzer()
        movies = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/imdb.csv"])
        fields = ["title", "description"]
        training_set = movies.split("genre", lambda v: v.split(","))
        self._classifier = in3120.NearestNeighborClassifier(training_set, fields, analyzer)

    def test_predict_movie_genre(self):
        results = list(self._classifier.classify("It was a dark and stormy night"))
        self.assertEqual("Thriller", results[0].category)

    def test_scores_are_sorted_descending(self):
        for voting in ("simple", "weighted"):
            options = in3120.NearestNeighborClassifier.Options(voting=voting)
            results = list(self._classifier.classify("It was a dark and stormy night", options))
            scores = [result.score for result in results]
            self.assertEqual(5, len(scores))
            self.assertListEqual(scores, sorted(scores, reverse=True))

    def test_barfs_on_invalid_options(self):
        options = in3120.NearestNeighborClassifier.Options(voting="dfdsf")
        with self.assertRaises(AssertionError):
            list(self._classifier.classify("It was a dark and stormy night", options))


if __name__ == '__main__':
    unittest.main(verbosity=2)
