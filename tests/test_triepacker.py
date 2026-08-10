# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
import tempfile
import os
from typing import Tuple, List, Any
from collections import Counter
from context import in3120


class TestTriePacker(unittest.TestCase):

    def setUp(self):
        self._figure_1d_from_tpt_paper = [("", 20), ("a", 13), ("aa", 10), ("ab", 3), ("b", 7)]
        self._strings_with_optional_meta = [("abba", 12), ("ab", None), ("abo", 99), ("aleksander", None), ("foo", None)]
        self._strings_with_emojis = [("💙", 0), ("💙⭐", 1), ("🔫💙🔫", 0)]

    def _pack_and_read(self, strings: Tuple[str, None | int], explain: bool) -> None | List[in3120.TriePacker.PackedCell]:
        analyzer = in3120.DummyAnalyzer()
        trie = in3120.SimpleTrie.from_strings2(strings, analyzer)
        with tempfile.NamedTemporaryFile(delete=True, prefix="tpt-", suffix=".bin") as file:
            filename = file.name
        try:
            packer = in3120.TriePacker()
            explanation = packer.pack(trie, filename, explain)
            if explanation is not None:
                self.assertEqual(0, explanation[0].offset)
                self.assertTrue(all(explanation[i].offset < explanation[i + 1].offset for i in range(len(explanation) - 1)))
                def receiver(node: in3120.TriePacker.PackedNode) -> None:
                    self.assertTrue(0 <= node.offset <= explanation[-1].offset)
                    for _, offset in node.children:
                        self.assertTrue(0 <= offset <= explanation[-1].offset)
                packer.dump(filename, receiver)
            return explanation
        finally:
            os.unlink(filename)

    def _verify_cell(self, cell: in3120.TriePacker.PackedCell, offset: int, value: Any, description: str) -> None:
        self.assertEqual(cell.offset, offset)
        self.assertEqual(cell.value, value)
        self.assertEqual(cell.description, description)

    def test_example_from_tpt_paper(self):
        explanation = self._pack_and_read(self._figure_1d_from_tpt_paper, True)
        self.assertEqual(19, len(explanation))
        self._verify_cell(explanation[0],  0,  20,         "Offset of root node")
        self._verify_cell(explanation[1],  8,  (10, True), "Node value of 'aa' plus finality bit")
        self._verify_cell(explanation[2],  9,  0,          "Size of index to child nodes of 'aa' in bytes")
        self._verify_cell(explanation[3],  10, (3, True),  "Node value of 'ab' plus finality bit")
        self._verify_cell(explanation[4],  11, 0,          "Size of index to child nodes of 'ab' in bytes")
        self._verify_cell(explanation[5],  12, (13, True), "Node value of 'a' plus finality bit")
        self._verify_cell(explanation[6],  13, 4,          "Size of index to child nodes of 'a' in bytes")
        self._verify_cell(explanation[7],  14, "a",        "Index key for 'aa' coming from 'a'")
        self._verify_cell(explanation[8],  15, 4,          "Relative offset of node 'aa' (12 - 4 = 8)")
        self._verify_cell(explanation[9],  16, "b",        "Index key for 'ab' coming from 'a'")
        self._verify_cell(explanation[10], 17, 2,          "Relative offset of node 'ab' (12 - 2 = 10)")
        self._verify_cell(explanation[11], 18, (7, True),  "Node value of 'b' plus finality bit")
        self._verify_cell(explanation[12], 19, 0,          "Size of index to child nodes of 'b' in bytes")
        self._verify_cell(explanation[13], 20, (20, True), "Node value of '' plus finality bit")
        self._verify_cell(explanation[14], 21, 4,          "Size of index to child nodes of '' in bytes")
        self._verify_cell(explanation[15], 22, "a",        "Index key for 'a' coming from ''")
        self._verify_cell(explanation[16], 23, 8,          "Relative offset of node 'a' (20 - 8 = 12)")
        self._verify_cell(explanation[17], 24, "b",        "Index key for 'b' coming from ''")
        self._verify_cell(explanation[18], 25, 2,          "Relative offset of node 'b' (20 - 2 = 18)")

    def test_explain_flag_works(self):
        self.assertIsNotNone(self._pack_and_read(self._figure_1d_from_tpt_paper, True))
        self.assertIsNone(self._pack_and_read(self._figure_1d_from_tpt_paper, False))

    def test_barfs_on_unsupported_meta(self):
        for unsupported in [-1, "grok"]:
            with self.assertRaises(AssertionError):
                self._pack_and_read(self._figure_1d_from_tpt_paper + [("barf", unsupported)], False)

    def test_handles_none_values_as_meta(self):
        explanation = self._pack_and_read(self._strings_with_optional_meta, True)
        count1 = sum(1 for cell in explanation if isinstance(cell.value, tuple) and cell.value[0] is not None)
        count2 = sum(1 for cell in explanation if isinstance(cell.value, tuple) and cell.value[0] is None and cell.value[1])
        self.assertEqual(count1, 2)
        self.assertEqual(count2, 3)

    def test_handles_empty_trie(self):
        explanation = self._pack_and_read([], True)
        values = [cell.value for cell in explanation]
        adjustment = explanation[1].offset
        expected = [1 + adjustment - 1, (None, False), 0]
        self.assertListEqual(values, expected)

    def test_handles_nonascii_symbols(self):
        explanation = self._pack_and_read(self._strings_with_emojis, True)
        counter = Counter(cell.value for cell in explanation if isinstance(cell.value, str))
        self.assertEqual(2, counter["💙"])
        self.assertEqual(2, counter["🔫"])
        self.assertEqual(1, counter["⭐"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
