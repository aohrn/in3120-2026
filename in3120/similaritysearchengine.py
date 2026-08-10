# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=not-callable

from dataclasses import dataclass
from typing import Iterator, Iterable
import faiss  # type: ignore[import-untyped]
from .embedder import Embedder
from .corpus import Corpus
from .document import Document
from .analyzer import Analyzer


class SimilaritySearchEngine:
    """
    A search engine based on doing approximate nearest neighbor (ANN) lookups
    over embedding vectors generated from a document corpus.

    This implementation relies on selected open-source libraries. For more on
    ANNs, see, e.g.:

    - https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6
    - https://wangzwhu.github.io/home/file/acmmm-t-part3-ann.pdf
    - https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/
    - ...and many more

    Uses spaCy (https://spacy.io/) to generate the embedding vectors for documents, using the
    precomputed word embeddings that come with the package. Much better embedding models exist,
    see https://huggingface.co/spaces/mteb/leaderboard for details. Some plausible open-source
    alternatives would have been, e.g.:

    - SentenceTransformers (https://sbert.net/)
    - transformers (https://github.com/huggingface/transformers)
    - UForm (https://github.com/unum-cloud/uform)
    - ...and many more

    Uses FAISS (https://github.com/facebookresearch/faiss) to realize the ANN index
    over the generated embedding vectors. Some plausible open-source alternatives would
    have been, e.g.:

    - Annoy (https://github.com/spotify/annoy)
    - FALCONN (https://github.com/falconn-lib/falconn)
    - NMSLIB (https://github.com/nmslib/nmslib)
    - Chroma (https://www.trychroma.com/)
    - DiskANN (https://github.com/microsoft/DiskANN/)
    - Postgres with the pgvector extension (https://github.com/pgvector/pgvector)
    - SQLite with the sqlite-vss extension (https://github.com/asg017/sqlite-vss)
    - ScaNN (https://github.com/google-research/google-research/tree/master/scann)
    - Any search engine based on Lucene (https://lucene.apache.org/), such as
      Elasticsearch (https://www.elastic.co/search-labs/vector-search-elasticsearch-rationale).
    - USearch (https://github.com/unum-cloud/usearch)
    - NGT (https://github.com/yahoojapan/NGT)
    - Milvus (https://milvus.io/en/)
    - Weaviate (https://weaviate.io/)
    - Vespa (https://vespa.ai/)
    - Pinecone (https://www.pinecone.io/)
    - Qdrant (https://qdrant.tech/)
    - Zilliz (https://zilliz.com/)
    - Vald (https://vald.vdaas.org/)
    - ...and many more
    """

    @dataclass
    class Options:
        """
        Query-time options. Controls lookup behavior.
        """
        hit_count: int = 5  # The maximum number of results we will emit.

    @dataclass
    class Result:
        """
        An individual lookup result, as reported back to the client.
        """
        score: float        # The relevance score, i.e., the distance between the query and the document. 
        document: Document  # The document with the matching content.

    def __init__(self, corpus: Corpus, fields: Iterable[str], analyzer: Analyzer):

        # FAISS barfs on an empty corpus.
        assert len(corpus or []) > 0

        # The documents to index, plus basic helpers.
        self._corpus = corpus
        self._embedder = Embedder(analyzer)

        # Place the normalized documents in embedding space. Normalize the embeddings.
        # For large corpora, we could embed/add data in batches.
        embeddings = self._embedder.to_matrix([self._embedder.from_document(d, fields) for d in corpus])
        faiss.normalize_L2(embeddings)

        # Enables us to map from matrix row indices to document identifiers. This gives us some robustness
        # in case document identifiers should become, e.g., arbitrary GUIDs. If the N document identifiers
        # are all integers {0, 1, ..., N - 1} then this is superfluous but benign.
        self._mappings = [d.document_id for d in self._corpus]

        # The ANN index. See https://github.com/facebookresearch/faiss/wiki/The-index-factory for options.
        dimensionality = embeddings[0].shape[0]
        self._index = faiss.index_factory(dimensionality, "Flat", faiss.METRIC_INNER_PRODUCT)
        self._index.train(embeddings)
        self._index.add(embeddings)

        # Sanity checks.
        assert self._index.is_trained
        assert self._index.ntotal == self._corpus.size()

    def evaluate(self, query: str, options: Options | None = None) -> Iterator[Result]:
        """
        Consults the ANN index to locate the documents that are the closest to the
        given query in embedding space.

        The best-matching documents are yielded back to the client.
        """
        # Default options apply unless specified.
        options = options or self.Options()

        # Empty query? Ignore leading/trailing whitespace.
        query = query.strip() if query else ""
        if not query:
            return

        # Place the normalized query string in embedding space. Normalize the embedding.
        # FAISS allows us to specify multiple query vectors to be evaluated in one swoop,
        # but we just have a single query.
        embedding = self._embedder.to_matrix([self._embedder.from_buffer(query)])
        faiss.normalize_L2(embedding)

        # Lookup! See, e.g., https://github.com/facebookresearch/faiss/wiki/Faster-search for options.
        distances, indices = self._index.search(embedding, min(100, max(1, options.hit_count)))

        # With METRIC_INNER_PRODUCT as our metric and normalized vectors, the emitted scores are cosine
        # similarity scores and are emitted back in descending order. With another metric where scores
        # would be distances and emitted back in ascending order, we might want to negate the scores
        # before emitting them in order to keep to the convention that "<" for scores means "ranks below".
        # See, e.g., https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances for more.
        for distance, index in zip(distances[0], indices[0]):
            yield self.Result(float(distance), self._corpus[self._mappings[index]])
