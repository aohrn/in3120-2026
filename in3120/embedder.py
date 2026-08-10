# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from typing import Iterable, Sequence
import spacy
import numpy as np
from .analyzer import Analyzer
from .document import Document


class Embedder:
    """
    Utility class for creating an embedding from a document or text buffer. An embedding is a dense vector
    representation of the input text. The dimensionality of the vector space is fixed and pre-determined,
    each dimension of this embedding space being mathematically abstract and hard to assign a human interpretation.
    If two embeddings are close to each other in the embedding space, then the texts they represent are similar.

    Uses spaCy (https://spacy.io/) to generate the embeddings, using the precomputed word embeddings that
    come with the package. Much better embedding models exist, see https://huggingface.co/spaces/mteb/leaderboard
    for details. Some plausible open-source alternatives would have been, e.g.:

    - SentenceTransformers (https://sbert.net/)
    - transformers (https://github.com/huggingface/transformers)
    - UForm (https://github.com/unum-cloud/uform)
    """

    # Shared across instances, initialized on demand below.
    _NLP: spacy.language.Language | None = None

    def __init__(self, analyzer: Analyzer):
        # Preprocessor, comes in addition to what the open-source embedding library does.
        self._analyzer = analyzer

        # The machinery for generating embedding vectors from text buffers. Assume English.
        if Embedder._NLP is None:
            Embedder._NLP = self._load_spacy("en_core_web_md")

    def _load_spacy(self, model: str) -> spacy.language.Language:
        """
        Loads the spaCy model (i.e, text-processing pipeline) to use. This is loaded once per process, usually.
        See, e.g., https://spacy.io/models and https://github.com/explosion/spacy-models for details.

        We're not really interested in the pipeline components here, and hence exclude them to speed things
        up when generating embeddings. Rather, we are interested in using the word vector tables that come with
        the 'md' and 'lg' size models.
        """
        try:
            return spacy.load(model, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        except (OSError, AttributeError) as exception:
            raise IOError(f"Do 'python -m spacy download {model}'.") from exception

    def from_buffer(self, buffer: str, analyze: bool = True) -> np.ndarray:
        """
        Generates the embedding vector representation of the given buffer. Unless otherwise specified, the
        input buffer is preprocessed by the analyzer prior to being embedded. The generated embedding is not
        normalized to unit length.

        The current implementation simply returns the averaged word vector, computed over all words in the given
        buffer. A more elaborate implementation could do a weighted average, with the weights being, e.g., the
        TF-IDF scores of each word.
        """
        assert Embedder._NLP is not None
        buffer = self._analyzer.join(buffer) if analyze else buffer
        return np.array(Embedder._NLP(buffer).vector.data)  # pylint: disable=not-callable

    def from_document(self, document: Document, fields: Iterable[str]) -> np.ndarray:
        """
        Creates the embedding vector representation of the given document, only considering the given set of
        named document fields.
        """
        return self.from_buffer(" \0 ".join(self._analyzer.join(document.get_field(f, "")) for f in fields), False)

    def to_matrix(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        """
        Given a list of row vectors, stacks them vertically into a matrix.
        """
        return np.vstack(embeddings)
