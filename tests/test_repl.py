# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

import contextlib
import unittest
import requests
import repl
from dataclasses import dataclass


class TestREPL(unittest.TestCase):

    def test_repls_have_docstrings(self):
        for function in repl.REPLS.values():
            self.assertIsNotNone(function.__doc__)

    def test_repls_start_up_without_barfing(self):
        flag = repl.IS_INTERACTIVE
        repl.IS_INTERACTIVE = False  # Don't block waiting on user input.
        with contextlib.redirect_stdout(None):
            for function in repl.REPLS.values():
                function()
        repl.IS_INTERACTIVE = flag

    def test_simple_ajax_server(self):
        with contextlib.redirect_stdout(None):
            def evaluator(query: str):
                return [{"foo": query.lower()}, {"bar": query.upper()}]
            httpd = repl.create_ajax_server(evaluator)
            try:
                response = requests.get(f"http://localhost:{httpd.server_address[1]}/query?q=TeSt", timeout=3)
                self.assertEqual(response.status_code, 200)
                json = response.json()
                self.assertTrue("duration" in json)
                self.assertTrue(isinstance(json["duration"], float))
                self.assertTrue("matches" in json)
                self.assertListEqual(json["matches"], [{"foo": "test"}, {"bar": "TEST"}])
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_conversion_to_plain_python(self):
        @dataclass
        class Foo1:
            a: int
            b: str
        class Foo2:
            def to_dict(self):
                return {"banana": True}
        foo1 = Foo1(1, "foo")
        foo2 = Foo2()
        self.assertEqual(repl.to_plain_python(21), 21)
        self.assertEqual(repl.to_plain_python("apple"), "apple")
        self.assertDictEqual(repl.to_plain_python(foo1), {"a": 1, "b": "foo"})
        self.assertDictEqual(repl.to_plain_python(foo2), {"banana": True})
        self.assertListEqual(repl.to_plain_python([foo1, foo2]), [{"a": 1, "b": "foo"}, {"banana": True}])
        self.assertDictEqual(repl.to_plain_python({"foo": foo1}), {"foo": {"a": 1, "b": "foo"}})


if __name__ == '__main__':
    unittest.main(verbosity=2)
