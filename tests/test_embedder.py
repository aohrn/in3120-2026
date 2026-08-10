# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
import numpy.testing as npt
from context import in3120


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        analyzer = in3120.SimpleAnalyzer()
        self._embedder = in3120.Embedder(analyzer)

    def test_dimensionality_embedding(self):
        for analyze in (True, False):
            embedding = self._embedder.from_buffer("Hello world", analyze)
            self.assertEqual(embedding.shape, (300,))

    def test_dimensionality_matrix(self):
        buffers = ["Hello world", "The banana was yellow", "Saft suse"]
        matrix = self._embedder.to_matrix([self._embedder.from_buffer(buffer) for buffer in buffers])
        self.assertEqual(matrix.shape, (3, 300))

    def test_embed_buffer_and_document_produce_equal_result(self):
        buffer = "Hello world"
        document = in3120.InMemoryDocument(0, {"body": buffer})
        embedding1 = self._embedder.from_buffer(buffer)
        embedding2 = self._embedder.from_document(document, ["body"])
        npt.assert_array_equal(embedding1, embedding2)

    def test_preprocessing_can_be_controlled(self):
        class TokenFilter(in3120.Normalizer):
            def normalize(self, token: str) -> str:
                return "" if token in {"world"} else token
        analyzer = in3120.Analyzer(in3120.ChainedNormalizer([in3120.SimpleNormalizer(), TokenFilter()]), in3120.SimpleTokenizer())
        embedder = in3120.Embedder(analyzer)
        buffer = "Hello world"
        embedding1 = embedder.from_buffer(buffer, True)
        embedding2 = embedder.from_buffer(buffer, False)
        npt.assert_raises(AssertionError, npt.assert_array_equal, embedding1, embedding2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
