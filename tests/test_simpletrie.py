# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pyright: reportOptionalMemberAccess=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false

import unittest
from random import choice
from context import in3120


class TestSimpleTrie(unittest.TestCase):

    def setUp(self):
        self._analyzer = in3120.SimpleAnalyzer()
        self._root = in3120.SimpleTrie.from_strings(["abba", "ØRRET", "abb", "abbab", "abbor"], self._analyzer)

    def test_consume_and_final(self):
        root = self._root
        self.assertTrue(not root.is_final())
        self.assertIsNone(root.consume("snegle"))
        node = root["ab"]
        self.assertTrue(not node.is_final())
        node = node.consume("b")
        node = node.consume("")
        self.assertTrue(node.is_final())
        self.assertEqual(node, root.consume("abb"))

    def test_containment(self):
        self.assertTrue("ørret" in self._root)
        self.assertFalse("ørr" in self._root)
        self.assertTrue("abbor" in self._root)
        self.assertFalse("abborrrr" in self._root)
        child = self._root.child("a")
        self.assertTrue("bbor" in child)

    def test_dump_strings(self):
        root = in3120.SimpleTrie.from_strings(["elle", "eller", "ELLEN", "hurra   FOR deg"], self._analyzer)
        self.assertListEqual(list(root.strings()), ["elle", "ellen", "eller", "hurra for deg"])
        node = root.consume("el")
        self.assertListEqual(list(node.strings()), ["le", "len", "ler"])
        self.assertListEqual(list(node), ["le", "len", "ler"])

    def test_add_is_idempotent(self):
        root = in3120.SimpleTrie.from_strings(["abba", "ABBA", "ABBA", "abBa"], self._analyzer)
        self.assertListEqual(list(root.strings()), ["abba"])

    def test_transitions_sorted(self):
        root = self._root
        self.assertListEqual(list(root.transitions()), ["a", "ø"])
        node = root.consume("abb")
        self.assertListEqual(list(node.transitions()), ["a", "o"])
        node = node.consume("o")
        self.assertListEqual(list(node.transitions()), ["r"])
        node = node.consume("r")
        self.assertListEqual(list(node.transitions()), [])

    def test_transitions_unsorted(self):
        root = in3120.SimpleTrie()
        for _ in range(100):
            root.add(["".join(choice("abcdefghijklmnopqrstuvwxyz") for _ in range(20))], self._analyzer)
        transitions1 = list(root.transitions(False))
        transitions2 = list(root.transitions(True))
        self.assertFalse(transitions1 == transitions2)
        self.assertListEqual(sorted(transitions1), sorted(transitions2))

    def test_child(self):
        root = self._root
        self.assertIsNotNone(root.child("a"))
        self.assertIsNone(root.child("ab"))
        child = root.child("a")
        child = child.child("b")
        child = child.child("b")
        self.assertIsNone(child.child(""))

    def test_with_meta_data(self):
        root = in3120.SimpleTrie.from_strings2([("aleksander", 2104), ("julaften", 2412), ("nei", None)], self._analyzer)
        self.assertFalse(root.has_meta())
        self.assertFalse(root.consume("aleks").is_final())
        self.assertFalse(root.consume("aleks").has_meta())
        self.assertIsNone(root.consume("aleks").get_meta())
        self.assertTrue(root.consume("aleksander").is_final())
        self.assertTrue(root.consume("aleksander").has_meta())
        self.assertEqual(root.consume("aleksander").get_meta(), 2104)
        self.assertTrue(root.consume("julaften").is_final())
        self.assertTrue(root.consume("julaften").has_meta())
        self.assertEqual(root.consume("julaften").get_meta(), 2412)
        self.assertTrue(root.consume("nei").is_final())
        self.assertFalse(root.consume("nei").has_meta())
        self.assertIsNone(root.consume("nei").get_meta())

    def test_counts(self):
        root = in3120.SimpleTrie.from_strings2([("aleks", 2104), ("jul", 2412), ("alibaba", 1001), ("nei", None)], self._analyzer)
        counts = root.counts()
        self.assertEqual(counts.node, 17)
        self.assertEqual(counts.unique, 17)
        self.assertEqual(counts.final, 4)
        self.assertEqual(counts.meta, 3)

    def test_to_dot(self):
        root = in3120.SimpleTrie.from_strings2([("aleks", 2104), ("jul", 2412), ("alibaba", 1001), ("nei", None)], self._analyzer)
        dot = root.to_dot()
        self.assertIsNotNone(dot)
        self.assertTrue(dot.startswith("digraph {"))
        self.assertEqual(dot.count("shape="), 17)
        self.assertEqual(dot.count("shape=doublecircle"), 4)
        self.assertEqual(dot.count("shape=circle"), 13)
        self.assertEqual(dot.count("label=\"2104\""), 1)
        self.assertEqual(dot.count("label=\"2412\""), 1)
        self.assertEqual(dot.count("label=\"1001\""), 1)
        self.assertEqual(dot.count("->"), 16)
        self.assertTrue(dot.endswith("}"))

    def test_add_is_idempotent_unless_meta_data_differs(self):
        root = in3120.SimpleTrie.from_strings2([("abba", 74), ("abba", 74)], self._analyzer)
        self.assertListEqual(list(root.strings()), ["abba"])
        self.assertEqual(root.consume("abba").get_meta(), 74)
        root = in3120.SimpleTrie()
        with self.assertRaises(AssertionError):
            root.add2([("abba", 74), ("abba", 99)], self._analyzer)

    def test_barfs_on_none(self):
        root = in3120.SimpleTrie()
        with self.assertRaises(AssertionError):
            root.add2([(None, 74)], self._analyzer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
