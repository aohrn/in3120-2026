# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

from typing import List
import sys
import unittest
import tests


def build_test_suite(test_cases: List[str]) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for test_case in ["TestGeneral"] + test_cases:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(tests, test_case)))
    return suite


def assignment_a_suite() -> unittest.TestSuite:
    return build_test_suite(["TestInMemoryInvertedIndexWithoutCompression",
                             "TestPostingsMerger", "TestBooleanSearchEngine"])


def assignment_b_1_suite() -> unittest.TestSuite:
    return build_test_suite(["TestSuffixArray", "TestSimpleTrie", "TestStringFinder"])


def assignment_b_2_suite() -> unittest.TestSuite:
    return build_test_suite(["TestEditTable", "TestEditSearchEngine"])


def assignment_c_1_suite() -> unittest.TestSuite:
    return build_test_suite(["TestSimpleSearchEngine"])


def assignment_c_2_suite() -> unittest.TestSuite:
    return build_test_suite(["TestExtendedBooleanSearchEngine"])


def assignment_d_1_suite() -> unittest.TestSuite:
    return build_test_suite(["TestBetterRanker", "TestShingleGenerator", "TestWordShingleGenerator", "TestSparseDocumentVector"])


def assignment_e_1_suite() -> unittest.TestSuite:
    return build_test_suite(["TestNaiveBayesClassifier"])


def assignment_x_suite() -> unittest.TestSuite:
    return build_test_suite(["TestSimpleNormalizer", "TestDummyNormalizer", "TestChainedNormalizer",
                             "TestSimpleTokenizer", "TestDummyTokenizer", "TestAnalyzer", "TestInMemoryDictionary",
                             "TestInMemoryDocument", "TestInMemoryCorpus", "TestCorpusLoader", "TestSieve", "TestVariableByteCodec",
                             "TestInMemoryPostingList", "TestCompressedInMemoryPostingList",
                             "TestInMemoryInvertedIndexWithCompression", "TestExpressionComposer",
                             "TestShallowCaseExtractor", "TestDocumentPipeline", "TestSimpleRanker",
                             "TestSoundexNormalizer", "TestPorterNormalizer",
                             "TestSimilaritySearchEngine", "TestEmbedder", "TestWildcardExpander",
                             "TestEliasGammaCodec", "TestBloomFilter", "TestVectorizer",
                             "TestDummyInMemoryInvertedIndex", "TestRocchioClassifier",
                             "TestWindowFinder", "TestNearestNeighborClassifier", "TestUnigramTokenizer",
                             "TestBinaryLogisticRegressionClassifier", "TestEvaluationMetrics",
                             "TestPageRank", "TestSimple9Codec", "TestFeatureSelector",
                             "TestTriePacker", "TestPackedTrie", "TestAcronymExtractor", "TestGradientBanditEngine",
                             "TestSimpleDAGTrie", "TestWordEditSearchEngine", "TestTREC"])


def test_framework_suite() -> unittest.TestSuite:
    return build_test_suite(["TestREPL"])


def main():
    mappings = {
        "a":   assignment_a_suite,
        "b-1": assignment_b_1_suite,
        "b-2": assignment_b_2_suite,
        "c-1": assignment_c_1_suite,
        "c-2": assignment_c_2_suite,
        "d-1": assignment_d_1_suite,
        "e-1": assignment_e_1_suite,
        "t":   test_framework_suite,
        "x":   assignment_x_suite,
    }
    targets = sys.argv[1:] or [key for key in mappings if key != "t"]
    suite = unittest.TestSuite()
    for target in targets:
        if target.lower() in mappings:
            suite.addTests(mappings[target.lower()]())
    if suite.countTestCases() > 0:
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)


if __name__ == "__main__":
    main()
