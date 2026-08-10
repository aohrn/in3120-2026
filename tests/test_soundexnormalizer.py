# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from context import in3120


class TestSoundexNormalizer(unittest.TestCase):

    def setUp(self):
        self._normalizer = in3120.SoundexNormalizer()

    def test_normalize_no_input(self):
        for argument in ["", None]:
            with self.assertRaises(ValueError):
                self._normalizer.normalize(argument)

    def test_normalize_weird_inputs(self):
        self.assertEqual(self._normalizer.normalize("!"), "!000")
        self.assertEqual(self._normalizer.normalize("øhRn"), "Ø650")
        self.assertEqual(self._normalizer.normalize("Bæ?"), "B000")

    def test_normalize_short_inputs(self):
        self.assertEqual(self._normalizer.normalize("A"), "A000")
        self.assertEqual(self._normalizer.normalize("thek"), "T200")

    def test_normalize_wikipedia_examples(self):
        self.assertEqual(self._normalizer.normalize("Robert"), "R163")
        self.assertEqual(self._normalizer.normalize("Rupert"), "R163")
        self.assertEqual(self._normalizer.normalize("Rubin"), "R150")
        self.assertEqual(self._normalizer.normalize("Ashcraft"), "A261")
        self.assertEqual(self._normalizer.normalize("Ashcroft"), "A261")
        self.assertEqual(self._normalizer.normalize("Tymczak"), "T520")
        self.assertEqual(self._normalizer.normalize("Pfister"), "P123")
        self.assertEqual(self._normalizer.normalize("Honeyman"), "H500")

    def test_normalize_codedrome_examples(self):
        self.assertEqual(self._normalizer.normalize("Johnson"), "J525")
        self.assertEqual(self._normalizer.normalize("Jonson"), "J525")
        self.assertEqual(self._normalizer.normalize("Adams"), "A352")
        self.assertEqual(self._normalizer.normalize("Addams"), "A352")
        self.assertEqual(self._normalizer.normalize("Davis"), "D120")
        self.assertEqual(self._normalizer.normalize("Davies"), "D120")
        self.assertEqual(self._normalizer.normalize("Simons"), "S520")
        self.assertEqual(self._normalizer.normalize("Simmons"), "S520")
        self.assertEqual(self._normalizer.normalize("Richards"), "R263")
        self.assertEqual(self._normalizer.normalize("Richardson"), "R263")
        self.assertEqual(self._normalizer.normalize("Taylor"), "T460")
        self.assertEqual(self._normalizer.normalize("Tailor"), "T460")
        self.assertEqual(self._normalizer.normalize("Carter"), "C636")
        self.assertEqual(self._normalizer.normalize("Chater"), "C360")
        self.assertEqual(self._normalizer.normalize("Stevenson"), "S315")
        self.assertEqual(self._normalizer.normalize("Stephenson"), "S315")
        self.assertEqual(self._normalizer.normalize("Stevenson"), "S315")
        self.assertEqual(self._normalizer.normalize("Naylor"), "N460")
        self.assertEqual(self._normalizer.normalize("Smith"), "S530")
        self.assertEqual(self._normalizer.normalize("Smythe"), "S530")
        self.assertEqual(self._normalizer.normalize("Harris"), "H620")
        self.assertEqual(self._normalizer.normalize("Harrys"), "H620")
        self.assertEqual(self._normalizer.normalize("Sim"), "S500")
        self.assertEqual(self._normalizer.normalize("Sym"), "S500")
        self.assertEqual(self._normalizer.normalize("Williams"), "W452")
        self.assertEqual(self._normalizer.normalize("Wilson"), "W425")
        self.assertEqual(self._normalizer.normalize("Baker"), "B260")
        self.assertEqual(self._normalizer.normalize("Barker"), "B626")
        self.assertEqual(self._normalizer.normalize("Wells"), "W420")
        self.assertEqual(self._normalizer.normalize("Wills"), "W420")
        self.assertEqual(self._normalizer.normalize("Fraser"), "F626")
        self.assertEqual(self._normalizer.normalize("Frazer"), "F626")
        self.assertEqual(self._normalizer.normalize("Jones"), "J520")
        self.assertEqual(self._normalizer.normalize("Johns"), "J520")
        self.assertEqual(self._normalizer.normalize("Wilks"), "W420")
        self.assertEqual(self._normalizer.normalize("Wilkinson"), "W425")
        self.assertEqual(self._normalizer.normalize("Hunt"), "H530")
        self.assertEqual(self._normalizer.normalize("Hunter"), "H536")
        self.assertEqual(self._normalizer.normalize("Sanders"), "S536")
        self.assertEqual(self._normalizer.normalize("Saunders"), "S536")
        self.assertEqual(self._normalizer.normalize("Parsons"), "P625")
        self.assertEqual(self._normalizer.normalize("Pearson"), "P625")
        self.assertEqual(self._normalizer.normalize("Robson"), "R125")
        self.assertEqual(self._normalizer.normalize("Robertson"), "R163")
        self.assertEqual(self._normalizer.normalize("Harker"), "H626")
        self.assertEqual(self._normalizer.normalize("Parker"), "P626")


if __name__ == '__main__':
    unittest.main(verbosity=2)
