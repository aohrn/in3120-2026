# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from types import GeneratorType
from timeit import default_timer as timer
from context import in3120


class TestStringFinder(unittest.TestCase):

    def setUp(self):
        self._analyzer = in3120.SimpleAnalyzer()

    def _simple_verify(self, finder, text, expected):
        results = list(finder.scan(text))
        self.assertListEqual([(result.surface, result.match) for result in results], expected)

    def _full_verify(self, finder, text, expected):
        results = list(finder.scan(text))
        self.assertEqual(len(results), len(expected))
        for r, e in zip(results, expected):
            self.assertEqual(r.match, e["match"])
            self.assertEqual(r.meta, e["meta"])
            self.assertEqual(r.surface, e["surface"])
            self.assertEqual(r.begin, e["span"][0])
            self.assertEqual(r.end, e["span"][1])

    def test_scan_matches_and_surface_forms_only(self):
        strings = ["romerike", "apple computer", "norsk", "norsk ørret", "sverige", "ørret", "banan", "a", "a b"]
        trie = in3120.SimpleTrie.from_strings(strings, self._analyzer)
        finder = in3120.StringFinder(trie, self._analyzer)
        self._simple_verify(finder, "en Norsk     ØRRET fra romerike likte abba fra Sverige",
                            [("Norsk", "norsk"), ("Norsk ØRRET", "norsk ørret"), ("ØRRET", "ørret"), ("romerike", "romerike"), ("Sverige", "sverige")])
        self._simple_verify(finder, "the apple is red", [])
        self._simple_verify(finder, "", [])
        self._simple_verify(finder, "apple computer banan foo sverige ben reddik fy fasan",
                            [("apple computer", "apple computer"), ("banan", "banan"), ("sverige", "sverige")])
        self._simple_verify(finder, "a a b", [("a", "a"), ("a", "a"), ("a b", "a b")])

    def test_scan_matches_and_spans(self):
        strings = ["eple", "drue", "appelsin", "drue appelsin rosin banan papaya"]
        trie = in3120.SimpleTrie.from_strings(strings, self._analyzer)
        finder = in3120.StringFinder(trie, self._analyzer)
        self._full_verify(finder, "et EPLE og en drue   appelsin  rosin banan papaya frukt",
                          [{'surface': 'EPLE', 'span': (3, 7), 'match': 'eple', 'meta': None},
                           {'surface': 'drue', 'span': (14, 18), 'match': 'drue', 'meta': None},
                           {'surface': 'appelsin', 'span': (21, 29), 'match': 'appelsin', 'meta': None},
                           {'surface': 'drue appelsin rosin banan papaya', 'span': (14, 49), 'match': 'drue appelsin rosin banan papaya', 'meta': None}])

    def test_with_phonetic_normalizer_and_meta(self):
        analyzer = in3120.Analyzer(in3120.SoundexNormalizer(), in3120.SimpleTokenizer())
        strings = ["Benedikt Richardson", "Smith"]
        trie = in3120.SimpleTrie.from_strings2(((x, x) for x in strings), analyzer)
        finder = in3120.StringFinder(trie, analyzer)
        self._full_verify(finder, "The Benedict  Richards and Smithe conjecture was proven false!",
                          [{'surface': 'Benedict Richards', 'span': (4, 22), 'match': 'B532 R263', 'meta': 'Benedikt Richardson'},
                           {'surface': 'Smithe', 'span': (27, 33), 'match': 'S530', 'meta': 'Smith'}])

    def test_uses_yield(self):
        trie = in3120.SimpleTrie.from_strings(["foo"], self._analyzer)
        finder = in3120.StringFinder(trie, self._analyzer)
        results = finder.scan("the foo bar")
        self.assertIsInstance(results, GeneratorType, "Are you using yield?")

    def test_mesh_terms_in_cran_corpus(self):
        mesh = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/mesh.txt"])
        cran = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/cran.xml"])
        trie = in3120.SimpleTrie.from_strings((d["body"] or "" for d in mesh), self._analyzer)
        finder = in3120.StringFinder(trie, self._analyzer)
        self._simple_verify(finder, cran[0]["body"], [("wing", "wing"), ("wing", "wing")])
        self._simple_verify(finder, cran[3]["body"], [("solutions", "solutions"), ("skin", "skin"), ("friction", "friction")])
        self._simple_verify(finder, cran[1254]["body"], [("electrons", "electrons"), ("ions", "ions")])

    def test_with_unigram_tokenizer_for_finding_arbitrary_substrings(self):
        analyzer = in3120.Analyzer(in3120.SimpleNormalizer(), in3120.UnigramTokenizer())
        trie = in3120.SimpleTrie.from_strings(["needle", "banana"], self._analyzer)
        finder = in3120.StringFinder(trie, analyzer)
        self._full_verify(finder, "thereisaneEdleinthishaystacksomewhereiamsureotherwisebananapineapple",
                          [{'surface': 'neEdle', 'span': (8, 14), 'match': 'needle', 'meta': None},
                           {'surface': 'banana', 'span': (53, 59), 'match': 'banana', 'meta': None}])

    def test_relative_insensitivity_to_dictionary_size(self):
        mesh = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/mesh.txt"])  # Contains more than 25K strings, including "medulla oblongata".
        trie1 = in3120.SimpleTrie.from_strings(["medulla oblongata"], self._analyzer)
        trie2 = in3120.SimpleTrie.from_strings((d["body"] or "" for d in mesh), self._analyzer)
        finder1 = in3120.StringFinder(trie1, self._analyzer)
        finder2 = in3120.StringFinder(trie2, self._analyzer)
        buffer = "The injury was located close to the medulla oblongata."
        finders = [finder1, finder2]
        times = [9999999, 9999999]
        for _ in range(10):
            for i in range(2):
                start = timer()
                results = list(finders[i].scan(buffer))
                end = timer()
                self.assertEqual(1, len(results))
                times[i] = min(times[i], end - start)
        ratio = times[1] / times[0]
        slack = 0.35  # Allow quite a bit of slack, to avoid spurious test failures.
        self.assertLessEqual(ratio, 1.0 + slack)
        self.assertLessEqual(1.0 - slack, ratio)


if __name__ == '__main__':
    unittest.main(verbosity=2)
