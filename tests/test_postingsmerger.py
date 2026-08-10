# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=line-too-long

import unittest
import types
import dis
from timeit import default_timer as timer
from context import in3120


class TestPostingsMerger(unittest.TestCase):

    def setUp(self):
        self._merger = in3120.PostingsMerger()

    def test_empty_lists_intersection(self):
        posting = in3120.Posting(123, 4)
        self.assertListEqual(list(self._merger.intersection(iter([]), iter([]))), [])
        self.assertListEqual(list(self._merger.intersection(iter([]), iter([posting]))), [])
        self.assertListEqual(list(self._merger.intersection(iter([posting]), iter([]))), [])

    def test_empty_lists_intersection_union(self):
        posting = in3120.Posting(123, 4)
        self.assertListEqual(list(self._merger.union(iter([]), iter([]))), [])
        self.assertListEqual([p.document_id for p in self._merger.union(iter([]), iter([posting]))], [posting.document_id])
        self.assertListEqual([p.document_id for p in self._merger.union(iter([posting]), iter([]))], [posting.document_id])

    def test_empty_lists_intersection_difference(self):
        posting = in3120.Posting(123, 4)
        self.assertListEqual(list(self._merger.difference(iter([]), iter([]))), [])
        self.assertListEqual([p.document_id for p in self._merger.difference(iter([posting]), iter([]))], [posting.document_id])
        self.assertListEqual([p.document_id for p in self._merger.difference(iter([]), iter([posting]))], [])

    def test_same_arguments_intersection(self):
        postings = [in3120.Posting(1, 0), in3120.Posting(2, 0), in3120.Posting(3, 0)]
        result = [p.document_id for p in self._merger.intersection(iter(postings), iter(postings))]
        self.assertListEqual(result, [1, 2, 3])

    def test_same_arguments_union(self):
        postings = [in3120.Posting(1, 0), in3120.Posting(2, 0), in3120.Posting(3, 0)]
        result = [p.document_id for p in self._merger.union(iter(postings), iter(postings))]
        self.assertListEqual(result, [1, 2, 3])

    def test_same_arguments_difference(self):
        postings = [in3120.Posting(1, 0), in3120.Posting(2, 0), in3120.Posting(3, 0)]
        result = [p.document_id for p in self._merger.difference(iter(postings), iter(postings))]
        self.assertListEqual(result, [])

    def test_order_independence_intersection(self):
        postings1 = [in3120.Posting(i, 0) for i in (1, 2, 3)]
        postings2 = [in3120.Posting(i, 0) for i in (2, 3, 6)]
        result12 = [p.document_id for p in self._merger.intersection(iter(postings1), iter(postings2))]
        result21 = [p.document_id for p in self._merger.intersection(iter(postings2), iter(postings1))]
        self.assertListEqual(result12, [2, 3])
        self.assertListEqual(result12, result21)

    def test_order_independence_union(self):
        postings1 = [in3120.Posting(i, 0) for i in (1, 2, 3)]
        postings2 = [in3120.Posting(i, 0) for i in (2, 3, 6)]
        result12 = [p.document_id for p in self._merger.union(iter(postings1), iter(postings2))]
        result21 = [p.document_id for p in self._merger.union(iter(postings2), iter(postings1))]
        self.assertListEqual(result12, [1, 2, 3, 6])
        self.assertListEqual(result12, result21)

    def test_order_dependence_difference(self):
        postings1 = [in3120.Posting(i, 0) for i in (1, 2, 3, 9)]
        postings2 = [in3120.Posting(i, 0) for i in (2, 3, 6, 8)]
        result12 = [p.document_id for p in self._merger.difference(iter(postings1), iter(postings2))]
        result21 = [p.document_id for p in self._merger.difference(iter(postings2), iter(postings1))]
        self.assertListEqual(result12, [1, 9])
        self.assertListEqual(result21, [6, 8])

    def test_ends_with_same_so_tail_is_empty(self):
        postings1 = [in3120.Posting(i, 0) for i in (1, 2, 6)]
        postings2 = [in3120.Posting(i, 0) for i in (2, 3, 6)]
        result1 = [p.document_id for p in self._merger.intersection(iter(postings1), iter(postings2))]
        result2 = [p.document_id for p in self._merger.union(iter(postings1), iter(postings2))]
        self.assertListEqual(result1, [2, 6])
        self.assertListEqual(result2, [1, 2, 3, 6])

    def test_uses_yield(self):
        postings1 = [in3120.Posting(i, 0) for i in (1, 2, 3)]
        postings2 = [in3120.Posting(i, 0) for i in (2, 3, 6)]
        result1 = self._merger.intersection(iter(postings1), iter(postings2))
        result2 = self._merger.union(iter(postings1), iter(postings2))
        result3 = self._merger.difference(iter(postings1), iter(postings2))
        for result in (result1, result2, result3):
            self.assertIsInstance(result, types.GeneratorType, "Are you using yield?")

    def _time_the_call(self, op, p1, p2) -> float:
        start = timer()
        _ = list(op(p1, p2))
        end = timer()
        return end - start

    def _exhibits_linear_behavior(self, op, message):
        # All posting list operations should be proportional to the length of
        # the posting lists. E.g., if processing two lists of length N takes
        # T seconds, then processing two lists having length 10N should take
        # approximately 10T seconds.
        length, factor, overlap = 500, 10, 25
        short_1 = [in3120.Posting(i, 0) for i in range(length)]
        short_2 = [in3120.Posting(i, 0) for i in range(length - overlap, 2 * length - overlap)]
        long_1 = [in3120.Posting(i, 0) for i in range(length * factor)]
        long_2 = [in3120.Posting(i, 0) for i in range((length - overlap) * factor, (2 * length - overlap) * factor)]
        pairs = [(short_1, short_2), (long_1, long_2)]
        times = [9999999, 9999999]
        for _ in range(5):
            for i in range(2):
                times[i] = min(times[i], self._time_the_call(op, iter(pairs[i][0]), iter(pairs[i][1])))
        ratio1 = times[1] / times[0]  # Observed timing ratio.
        ratio2 = ratio1 / factor  # Normalized timing ratio.
        slack = 0.40  # Allow quite a bit of slack, to avoid spurious test failures.
        self.assertLessEqual(ratio2, 1.0 + slack, message)
        self.assertLessEqual(1.0 - slack, ratio2, message)

    def test_linear_behavior_intersection(self):
        message = "Is your intersection code really linear in the length of the posting lists?"
        self._exhibits_linear_behavior(self._merger.intersection, message)

    def test_linear_behavior_union(self):
        message = "Is your union code really linear in the length of the posting lists?"
        self._exhibits_linear_behavior(self._merger.union, message)

    def test_linear_behavior_difference(self):
        message = "Is your difference code really linear in the length of the posting lists?"
        self._exhibits_linear_behavior(self._merger.difference, message)

    def _bytecode_smells_fishy(self, bytecode: dis.Bytecode) -> bool:
        # The posting lists are already sorted, and there should be no need to use
        # secondary data structures. This is a rather strict and conservative test.
        for instruction in bytecode:
            match instruction.opname:
                case "LOAD_CONST":
                    return any(canary in instruction.argrepr for canary in ("<listcomp>", "<dictcomp>", "<setcomp>"))
                case "LOAD_GLOBAL":
                    return any(canary in instruction.argrepr for canary in ("list", "dict", "set", "sorted"))
                case "BUILD_LIST" | "BUILD_MAP":
                    return True
        return False

    def test_does_not_use_unnecessary_constructs_intersection(self):
        message = "Are you resorting the posting lists, or using unnecessary secondary data structures in your intersection code?"
        self.assertFalse(self._bytecode_smells_fishy(dis.Bytecode(self._merger.intersection)), message)

    def test_does_not_use_unnecessary_constructs_union(self):
        message = "Are you resorting the posting lists, or using unnecessary secondary data structures in your union code?"
        self.assertFalse(self._bytecode_smells_fishy(dis.Bytecode(self._merger.union)), message)

    def test_does_not_use_unnecessary_constructs_difference(self):
        message = "Are you resorting the posting lists, or using unnecessary secondary data structures in your difference code?"
        self.assertFalse(self._bytecode_smells_fishy(dis.Bytecode(self._merger.difference)), message)

    def _process_query_with_two_terms(self, corpus, index, query, operator, expected):
        terms = list(index.get_terms(query))
        postings = [index[terms[i]] for i in range(len(terms))]
        self.assertEqual(len(postings), 2)
        merged = operator(postings[0], postings[1])
        documents = [corpus[posting.document_id] for posting in merged]
        self.assertEqual(len(documents), len(expected))
        self.assertListEqual([d.document_id for d in documents], expected)

    def _test_mesh_corpus(self, compressed: bool):
        analyzer = in3120.SimpleAnalyzer()
        corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/mesh.txt"])
        index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer, compressed)
        self._process_query_with_two_terms(corpus, index, "HIV  pROtein", self._merger.intersection,
                                           [11316, 11319, 11320, 11321])
        self._process_query_with_two_terms(corpus, index, "water Toxic", self._merger.union,
                                           [3078, 8138, 8635, 9379, 14472, 18572, 23234, 23985] +
                                           list(range(25265, 25282)))

    def test_uncompressed_mesh_corpus(self):
        self._test_mesh_corpus(False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
