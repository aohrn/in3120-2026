# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from typing import List, Tuple
from context import in3120


class TestAcronymExtractor(unittest.TestCase):

    def setUp(self):
        self._extractor = in3120.AcronymExtractor()

    def _verify_results(self, buffer: str, expected: List[Tuple[str, str]]):
        results = [(r.acronym, r.definition) for r in self._extractor.extract(buffer)]
        self.assertListEqual(results, expected)

    def test_basic_functionality_description_in_parenthesis(self):
        self._verify_results("The DNA (deoxyribonucleic acid) results just came in.", [("DNA", "deoxyribonucleic acid")])

    def test_basic_functionality_acronym_in_parenthesis(self):
        self._verify_results("Yikes, deoxyribonucleic acid (DNA) sure is a hard name to remember.", [("DNA", "deoxyribonucleic acid")])
        self._verify_results("People used The Microsoft Network (MSN).", [("MSN", "Microsoft Network")])

    def test_nothing_before_parenthesis(self):
        for prefix in ("", " "):
            self._verify_results(f"{prefix}(Reuters) The sun is shining in Oslo today.", [])

    def test_nothing_inside_parenthesis(self):
        for inside in ("", " ", "-"):
            self._verify_results(f"Blah blah ({inside}) blah?", [])

    def test_failed_alignment(self):
        self._verify_results("The Russian invasion of Ukraine (timeline)", [])
        self._verify_results("The copyright (C) symbol", [])
        self._verify_results("On paper, it's a bit more modern (and rounded) compared to what came before.", [])

    def test_require_space_before_parenthesis(self):
        self._verify_results("The linear regression of actual log(RT) was a strange thing.", [])

    def test_acronyms_with_special_characters(self):
        self._verify_results("Its predecessor's former name, the American Telephone and Telegraph Company (AT&T), is an...", [("AT&T", "American Telephone and Telegraph Company")])
        self._verify_results("Det var Åge Øivind (ÅØ) som ...", [("ÅØ", "Åge Øivind")])

    def test_definitions_with_special_characters(self):
        self._verify_results("I am a Regional Environmental Health & Safety (EHS) Manager.", [("EHS", "Environmental Health & Safety")])
        self._verify_results("The U.S. Citizenship and Immigration Services (USCIS) is proposing to...", [("USCIS", "U.S. Citizenship and Immigration Services")])
        self._verify_results("The Biden administration's handling of the Broadband Equity, Access, and Deployment (BEAD) program.", [("BEAD", "Broadband Equity, Access, and Deployment")])
        self._verify_results("Det var en bølle bak skuret (BBS) som..", [("BBS", "bølle bak skuret")])
        self._verify_results("Does text-to-speech (TTS) work well?", [("TTS", "text-to-speech")])

    def test_aligned_definitions_that_start_with_a_hyphen(self):
        self._verify_results("We have a fungible token (FT) and non-functioning dishwasher (FD)", [("FT", "fungible token")])

    def test_examples_from_schwarz_and_hearst_paper(self):
        buffer = (
            "Many abbreviations in biomedical text follow a predictable pattern, where the first letter of each word "
            "in the long form corresponds to one letter in the short form, as in methyl methanesulfonate sulfate (MMS). "
            "However, some cases require skipping words in the long form or matching internal letters, "
            "as in Gcn5-related N-acetyltransferase (GNAT)."
        )
        self._verify_results(buffer, [("MMS", "methyl methanesulfonate sulfate"), ("GNAT", "Gcn5-related N-acetyltransferase")])
        self._verify_results("Heat shock transcription factor (HSF)?", [("HSF", "Heat shock transcription factor")])
        self._verify_results("HSF (Heat shock transcription factor)?", [("HSF", "Heat shock transcription factor")])

    def test_examples_from_github(self):
        # Lifted from https://github.com/philgooch/abbreviation-extraction/blob/develop/test/features/schwartz_hearst.feature.
        buffer = (
            "The endoplasmic reticulum (ER) in Saccharomyces cerevisiae consists of a "
            "reticulum underlying the plasma membrane (cortical ER) and ER associated with "
            "the nuclear envelope (nuclear ER). "
            "The SH3 domain of Myo5p regulates the polymerization of actin through interactions with both Las17p, a homolog of "
            "mammalian Wiskott-Aldrich syndrome protein (WASP), and Vrp1p, a homolog of WASP-interacting protein (WIP). "
            "Ribonuclease P (RNase P) is a ubiquitous endoribonuclease that cleaves precursor tRNAs to generate mature 5prime prime or minute termini. "
            "The purified proteins were separated by sodium dodecyl sulfate-polyacrylamide gel electrophoresis (SDS-PAGE) and "
            "identified by peptide mass fingerprint analysis using matrix-assisted laser desorption/ionization (MALDI) mass spectrometry. "
            "Theory of mind (ToM; Smith 2009) broadly refers to humans' ability to represent the mental states of others, including their desires, beliefs, and intentions. "
            "We review astronomy and physics engagement with the Open Researcher and Contributor iD (ORCID) service as a solution. "
            'The Proceeds of Crime Act 2002 ("PoCA 2002"). '
            "Set up its own space monitoring center, called the Air and Space Operations Center (ASOC) in September 2020. "
            "...satellites, and the first Taikonaut (astronaut) in October. "
            "It is in our hands (and brains) to transform these ideas."
        )
        expected = [
            ("ER", "endoplasmic reticulum"),
            ("WASP", "Wiskott-Aldrich syndrome protein"),
            ("WIP", "WASP-interacting protein"),
            # ("RNase P", "Ribonuclease P"),
            ("SDS-PAGE", "sodium dodecyl sulfate-polyacrylamide gel electrophoresis"),
            ("MALDI", "matrix-assisted laser desorption/ionization"),
            # ("ToM", "Theory of mind"),
            ("ORCID", "Open Researcher and Contributor iD"),
            # ("PoCA 2002", "Proceeds of Crime Act 2002"),
            ("ASOC", "Air and Space Operations Center"),
        ]
        self._verify_results(buffer, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
