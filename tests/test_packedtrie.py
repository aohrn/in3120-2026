# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pyright: reportOptionalMemberAccess=false
# pyright: reportArgumentType=false

import unittest
import tempfile
import os
from context import in3120


class TestPackedTrie(unittest.TestCase):

    def setUp(self):
        self._strings = [("abba", 12), ("ab", None), ("abo", 99), ("aleksander", None), ("foo", None)]
        analyzer = in3120.SimpleAnalyzer()
        trie = in3120.SimpleTrie.from_strings2(self._strings, analyzer)
        with tempfile.NamedTemporaryFile(delete=True, prefix="tpt-", suffix=".bin") as file:
            self._filename = file.name
        packer = in3120.TriePacker()
        packer.pack(trie, self._filename)

    def tearDown(self):
        os.unlink(self._filename)

    def test_dump_strings(self):
        with in3120.PackedTrie(self._filename) as root:
            strings = list(root.strings())
            self.assertListEqual(strings, sorted(s for s, _ in self._strings))

    def test_counts(self):
        with in3120.PackedTrie(self._filename) as root:
            counts = root.counts()
            self.assertEqual(counts.node, 18)
            self.assertEqual(counts.final, 5)
            self.assertEqual(counts.meta, 2)

    def test_consume_and_check(self):
        with in3120.PackedTrie(self._filename) as root:
            for s, meta in self._strings:
                for node in (root.consume(s), root[s]):
                    self.assertIsNotNone(node)
                    self.assertTrue(node.is_final())
                    self.assertEqual(meta, node.get_meta())
                    self.assertEqual(meta is not None, node.has_meta())
            for s in ("ørret", "abbor", "kveite"):
                self.assertIsNone(root.consume(s))
                with self.assertRaises(KeyError):
                    _ = root[s]
            for s in ("a", "aleksan", "fo"):
                for node in (root.consume(s), root[s]):
                    self.assertIsNotNone(node)
                    self.assertFalse(node.is_final())
                    self.assertEqual(None, node.get_meta())
                    self.assertFalse(node.has_meta())

    def test_containment(self):
        with in3120.PackedTrie(self._filename) as trie:
            for s, _ in self._strings:
                self.assertTrue(s in trie)
            for s in ("ørret", "abbor", "kveite"):
                self.assertFalse(s in trie)

    def test_child_and_transitions(self):
        with in3120.PackedTrie(self._filename) as root:
            self.assertListEqual(["a", "f"], list(root.transitions()))
            self.assertListEqual(["b", "l"], list(root.child("a").transitions()))
            self.assertListEqual(["b", "o"], list(root.child("a").child("b").transitions()))
            self.assertListEqual(["a"], list(root.child("a").child("b").child("b").transitions()))
            self.assertListEqual([], list(root.child("a").child("b").child("b").child("a").transitions()))

    def test_child_and_check(self):
        with in3120.PackedTrie(self._filename) as root:
            for s, _ in self._strings:
                node = root
                for i, c in enumerate(s):
                    node = node.child(c)
                    self.assertEqual(node.is_final(), (s[:(i + 1)], node.get_meta()) in self._strings)

    def test_barfs_if_misconfigured(self):
        for filename in (None, "", self._filename[::-1]):
            with self.assertRaises((RuntimeError, FileNotFoundError, OSError)):
                with in3120.PackedTrie(filename) as root:
                    self.assertIsNotNone(root)

    def test_repack_packed_trie(self):
        with in3120.PackedTrie(self._filename) as root:
            filename2 = self._filename + "2"
            packer = in3120.TriePacker()
            explanation = packer.pack(root, filename2, True)
            self.assertEqual(71, len(explanation))
            self.assertEqual(os.path.getsize(self._filename), os.path.getsize(filename2))
            os.unlink(filename2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
