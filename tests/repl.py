# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=broad-exception-caught
# pylint: disable=too-few-public-methods

import json
import os
import pprint
import sys
import time
import threading
from dataclasses import asdict, is_dataclass
from timeit import default_timer as timer
from typing import Callable, Any
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote
from context import in3120


# For testing purposes, some global variables are exposed so that we can
# test-run the REPLs without blocking on interactive user input.
IS_INTERACTIVE = True
DEFAULT_QUERY = ""


# The colorama module is not part of the Python standard library.
# For self-containment reasons, make it optional and have output
# colorization reduce to a NOP if it's not installed.
try:
    from colorama import Fore, Style  # type: ignore[import-untyped]
except ImportError:
    print("Colorization disabled, 'pip install colorama' to enable.")
    class Fore:  # type: ignore[no-redef]
        GREEN = ""
        RED = ""
        LIGHTYELLOW_EX = ""
        LIGHTBLUE_EX = ""
    class Style:  # type: ignore[no-redef]
        RESET_ALL = ""


def data_path(filename: str) -> str:
    """
    Given the name of a file in the 'data' folder, returns its absolute path.
    """
    here = os.path.dirname(__file__)
    data = os.path.join(here, "..", "data")
    full = os.path.abspath(os.path.join(data, filename))
    return full


def to_plain_python(x: Any) -> Any:
    """
    Returns a plain Python object that is easily pretty-printed and serialized. 
    Converts dataclasses and custom objects as needed.
    """
    if is_dataclass(x):
        return asdict(x, dict_factory=lambda d: {k: v for k, v in d if v is not None})  # type: ignore[arg-type]
    if isinstance(x, list):
        return [to_plain_python(e) for e in x]
    if isinstance(x, dict):
        return {to_plain_python(k): to_plain_python(v) for k, v in x.items()}
    if isinstance(x, set):
        return set(to_plain_python(e) for e in x)
    if hasattr(x, 'to_dict') and callable(getattr(x, 'to_dict')):
        return x.to_dict()
    return x


def simple_repl(prompt: str, evaluator: Callable[[str], Any]) -> None:
    """
    Define a simple REPL to query from the terminal.
    """
    printer = pprint.PrettyPrinter()
    print(f"{Fore.LIGHTYELLOW_EX}Ctrl-C to exit.{Style.RESET_ALL}")
    try:
        while True:
            print(f"{Fore.GREEN}{prompt}>{Fore.RED}", end="")
            query = input() if IS_INTERACTIVE else DEFAULT_QUERY
            try:
                start = timer()
                matches = evaluator(query)
                end = timer()
                print(Fore.LIGHTBLUE_EX, end="")
                printer.pprint(to_plain_python(matches))
                print(f"{Fore.LIGHTYELLOW_EX}Evaluation took {end - start} seconds.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}{e.__class__.__name__}: {e}{Style.RESET_ALL}")
            if not IS_INTERACTIVE:
                break
    except KeyboardInterrupt:
        print(f"{Fore.LIGHTYELLOW_EX}\nBye!{Style.RESET_ALL}")


def simple_ajax(evaluator: Callable[[str], Any]) -> None:
    """
    Define a simple REPL to query from the browser.
    """
    httpd = create_ajax_server(evaluator)
    print(f"{Fore.LIGHTYELLOW_EX}Ctrl-C to exit.{Style.RESET_ALL}")
    try:
        print(f"{Fore.GREEN}Server running on localhost:{httpd.server_address[1]}, open your browser.{Style.RESET_ALL}", end="")
        if IS_INTERACTIVE:
            print(f"{Fore.RED}", end="")
            while True:
                _ = input()
    except KeyboardInterrupt:
        print(f"{Fore.LIGHTYELLOW_EX}\nBye!{Style.RESET_ALL}")
    finally:
        httpd.shutdown()
        httpd.server_close()


def create_ajax_server(evaluator: Callable[[str], Any]) -> ThreadingHTTPServer:
    """
    Creates, starts and returns a small HTTP server to query on a per keypress basis
    at localhost on a dynamically assigned port. Coordinated with index.html. The client
    needs to properly shut down the server afterwards.
    """
    class MyEncoder(json.JSONEncoder):
        """
        Custom JSON encoder, so that we can serialize custom IN3120 objects.
        """
        def default(self, o):
            return to_plain_python(o)
    class MyHandler(SimpleHTTPRequestHandler):
        """
        Custom HTTP handler. Supports GET only, suppresses all log messages.
        """
        def do_GET(self):
            if self.path.startswith("/query"):
                query = unquote(self.path.split('=')[1])
                start = timer()
                matches = evaluator(query)
                end = timer()
                results = {"duration": end - start, "matches": matches}
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(results, cls=MyEncoder).encode())
            else:
                super().do_GET()
        def log_message(self, format, *args):  # pylint: disable=redefined-builtin
            pass
    httpd = ThreadingHTTPServer(("", 0), MyHandler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    time.sleep(1)
    return httpd


def repl_a_1():
    """
    Enter one or more index terms and inspect their posting lists, using the Cranfield corpus.
    """
    print("Building inverted index from Cranfield corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("cran.xml")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    print(repl_a_1.__doc__.strip())
    simple_repl("terms", lambda ts: {t: list(index.get_postings_iterator(t)) for t in index.get_terms(ts)})


def repl_a_2():
    """
    Enter a complex Boolean query expression and find matching English names.
    """
    print("Building inverted index from English name corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("names.txt")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    engine = in3120.BooleanSearchEngine(corpus, index)
    options = in3120.BooleanSearchEngine.Options(optimize=True)
    print(repl_a_2.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    simple_repl("query", lambda e: list(engine.evaluate(e, options)))


def repl_b_1():
    """
    Enter a prefix phrase query and find matching Cranfield documents.
    """
    print("Building suffix array from Cranfield corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("cran.xml")])
    engine = in3120.SuffixArray(corpus, ["body"], analyzer)
    options = in3120.SuffixArray.Options(hit_count=5)
    print(repl_b_1.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print("Returned scores are occurrence counts.")
    simple_repl("query", lambda q: list(engine.evaluate(q, options)))


def repl_b_2():
    """
    Enter some text and locate words and phrases that are MeSH terms.
    """
    print("Building trie from MeSH corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("mesh.txt")])
    dictionary = in3120.SimpleTrie()
    dictionary.add((d["body"] for d in corpus), analyzer)
    engine = in3120.StringFinder(dictionary, analyzer)
    print(repl_b_2.__doc__.strip())
    simple_repl("text", lambda t: list(engine.scan(t)))


def repl_b_3():
    """
    Enter a prefix phrase query and find matching (non-closed) airports.
    """
    print("Building suffix array from airport corpus...")
    analyzer = in3120.SimpleAnalyzer()
    pipeline = in3120.DocumentPipeline([lambda d: d if d.get_field("type", "") != "closed" else None])
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("airports.csv")], None, pipeline)
    engine = in3120.SuffixArray(corpus, ["id", "type", "name", "iata_code"], analyzer)
    options = in3120.SuffixArray.Options(hit_count=5)
    print(repl_b_3.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print("Returned scores are occurrence counts.")
    simple_repl("query", lambda q: list(engine.evaluate(q, options)))


def repl_b_4():
    """
    Enter a prefix phrase query and find matching people in your browser as you type.
    """
    print("Building suffix array from Pantheon corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("pantheon.tsv")])
    engine = in3120.SuffixArray(corpus, ["name"], analyzer)
    options = in3120.SuffixArray.Options(hit_count=5)
    print(repl_b_4.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print("Returned scores are occurrence counts.")
    simple_ajax(lambda q: list(engine.evaluate(q, options)))


def repl_b_5():
    """
    Enter a query and find MeSH terms that are approximate matches.
    """
    print("Building trie from MeSH corpus...")
    analyzer = in3120.Analyzer(in3120.DummyNormalizer(), in3120.SimpleTokenizer())
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("mesh.txt")])
    dictionary = in3120.SimpleTrie()
    dictionary.add((d["body"] for d in corpus), analyzer)
    dictionary.add2([("shibboleth", "https://en.wikipedia.org/wiki/Shibboleth")], analyzer)
    engine = in3120.EditSearchEngine(dictionary, analyzer)
    options = in3120.EditSearchEngine.Options(hit_count=5, upper_bound=2)
    print(repl_b_5.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    simple_repl("query", lambda q: list(engine.evaluate(q, options)))


def repl_c_1():
    """
    Enter a query and find matching English news stories.
    """
    print("Indexing English news corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("en.txt")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    ranker = in3120.SimpleRanker()
    engine = in3120.SimpleSearchEngine(corpus, index)
    options = in3120.SimpleSearchEngine.Options(hit_count=5, match_threshold=0.5)
    print(repl_c_1.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print(f"Tokenizer is {analyzer.tokenizer.__class__.__name__}.")
    print(f"Ranker is {ranker.__class__.__name__}.")
    simple_repl("query", lambda q: list(engine.evaluate(q, ranker, options)))


def repl_c_2():
    """
    Enter an extended complex Boolean query expression and find matching English names.
    """
    print("Building inverted index from English name corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("names.txt")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    equivalences = ["aleksander", "alexander"]
    synonyms = in3120.SimpleTrie.from_strings2(((s, equivalences) for s in equivalences), analyzer)
    engine = in3120.ExtendedBooleanSearchEngine(corpus, index, synonyms)
    options = in3120.BooleanSearchEngine.Options(optimize=True)
    print(repl_c_2.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    simple_repl("query", lambda e: list(engine.evaluate(e, options)))


def repl_d_1():
    """
    Enter a query and find MeSH terms that approximately match.
    """
    print("Indexing MeSH corpus...")
    analyzer = in3120.Analyzer(in3120.SimpleNormalizer(), in3120.ShingleGenerator(3))
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("mesh.txt")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    ranker = in3120.SimpleRanker()
    engine = in3120.SimpleSearchEngine(corpus, index)
    options = in3120.SimpleSearchEngine.Options(hit_count=5, match_threshold=0.5)
    print(repl_d_1.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print(f"Normalizer is {analyzer.normalizer.__class__.__name__}.")
    print(f"Tokenizer is {analyzer.tokenizer.__class__.__name__}.")
    print(f"Ranker is {ranker.__class__.__name__}.")
    simple_repl("query", lambda q: list(engine.evaluate(q, ranker, options)))


def repl_d_2():
    """
    Enter a query and find matching English news stories, without stemming.
    """
    print("Indexing English news corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("en.txt")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    ranker = in3120.BetterRanker(corpus, index)
    engine = in3120.SimpleSearchEngine(corpus, index)
    options = in3120.SimpleSearchEngine.Options(hit_count=5, match_threshold=0.5)
    print(repl_d_2.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print(f"Normalizer is {analyzer.normalizer.__class__.__name__}.")
    print(f"Tokenizer is {analyzer.tokenizer.__class__.__name__}.")
    print(f"Ranker is {ranker.__class__.__name__}.")
    simple_repl("query", lambda q: list(engine.evaluate(q, ranker, options)))


def repl_d_3():
    """
    Enter a query and find matching English news stories, with stemming.
    """
    print("Indexing English news corpus...")
    analyzer = in3120.Analyzer(in3120.PorterNormalizer(), in3120.SimpleTokenizer())
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("en.txt")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    ranker = in3120.BetterRanker(corpus, index)
    engine = in3120.SimpleSearchEngine(corpus, index)
    options = in3120.SimpleSearchEngine.Options(hit_count=5, match_threshold=0.5)
    print(repl_d_3.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print(f"Normalizer is {analyzer.normalizer.__class__.__name__}.")
    print(f"Tokenizer is {analyzer.tokenizer.__class__.__name__}.")
    print(f"Ranker is {ranker.__class__.__name__}.")
    simple_repl("query", lambda q: list(engine.evaluate(q, ranker, options)))


def repl_d_4():
    """
    Enter a name and find phonetically similar English names.
    """
    print("Indexing randomly generated English names...")
    analyzer = in3120.Analyzer(in3120.SoundexNormalizer(), in3120.SimpleTokenizer())
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("names.txt")])
    index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    ranker = in3120.BetterRanker(corpus, index)
    engine = in3120.SimpleSearchEngine(corpus, index)
    options = in3120.SimpleSearchEngine.Options(hit_count=5, match_threshold=0.5)
    print(repl_d_4.__doc__.strip())
    simple_repl("query", lambda q: list(engine.evaluate(q, ranker, options)))


def repl_d_5():
    """
    Enter a query and find airports that match.
    """
    print("Indexing the set of airports in the world...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("airports.csv")])
    index = in3120.InMemoryInvertedIndex(corpus, ["id", "type", "name", "iata_code"], analyzer)
    ranker = in3120.BetterRanker(corpus, index)
    engine = in3120.SimpleSearchEngine(corpus, index)
    options = in3120.SimpleSearchEngine.Options(hit_count=5, match_threshold=0.5)
    print(repl_d_5.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    print(f"Normalizer is {analyzer.normalizer.__class__.__name__}.")
    print(f"Tokenizer is {analyzer.tokenizer.__class__.__name__}.")
    print(f"Ranker is {ranker.__class__.__name__}.")
    simple_repl("query", lambda q: list(engine.evaluate(q, ranker, options)))


def repl_e_1():
    """
    Enter some text and detect its language, using a naive Bayes classifier.
    """
    print("Initializing naive Bayes classifier from news corpora...")
    analyzer = in3120.SimpleAnalyzer()
    languages = ["en", "no", "da", "de"]
    training_set = {language: in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path(f"{language}.txt")]) for language in languages}
    classifier = in3120.NaiveBayesClassifier(training_set, ["body"], analyzer)
    print(repl_e_1.__doc__.strip())
    print(f"Known languages are {languages}.")
    print("Returned scores are log-probabilities.")
    simple_repl("text", lambda t: list(classifier.classify(t)))


def repl_e_2():
    """
    Enter a query and find approximately similar English news stories.
    """
    print("Indexing English news corpus using an ANN index...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("en.txt")])
    engine = in3120.SimilaritySearchEngine(corpus, ["body"], analyzer)
    options = in3120.SimilaritySearchEngine.Options(hit_count=5)
    print(repl_e_2.__doc__.strip())
    print(f"Lookup options are {to_plain_python(options)}.")
    simple_repl("query", lambda q: list(engine.evaluate(q, options)))


def repl_e_3():
    """
    Enter some text and have it classified as a movie genre, using a k-NN classifier.
    """
    print("Initializing k-NN classifier from movie corpus built over an ANN index...")
    analyzer = in3120.SimpleAnalyzer()
    movies = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("imdb.csv")])
    fields = ["title", "description"]
    training_set = movies.split("genre", lambda v: v.split(","))
    classifier = in3120.NearestNeighborClassifier(training_set, fields, analyzer)
    options = in3120.NearestNeighborClassifier.Options(k=3, voting="weighted")
    print(repl_e_3.__doc__.strip())
    print(f"Classification options are {to_plain_python(options)}.")
    simple_repl("text", lambda t: list(classifier.classify(t, options)))


def repl_x_1():
    """
    Enter some text and see what gets extracted based on simple case heuristics.
    """
    normalizer = in3120.SimpleNormalizer()
    extractor = in3120.ShallowCaseExtractor()
    options = in3120.ShallowCaseExtractor.Options()
    print(repl_x_1.__doc__.strip())
    print(f"Extraction options are {to_plain_python(options)}.")
    simple_repl("text", lambda t: list(extractor.extract(normalizer.canonicalize(t), options)))


def repl_x_2():
    """
    Enter some English text and see the Soundex codes.
    """
    normalizer = in3120.SoundexNormalizer()
    tokenizer = in3120.SimpleTokenizer()
    print(repl_x_2.__doc__.strip())
    simple_repl("text", lambda t: [normalizer.normalize(token) for token in tokenizer.strings(normalizer.canonicalize(t))])


def repl_x_3():
    """
    Enter some English text and see the stemmed terms.
    """
    normalizer = in3120.PorterNormalizer()
    tokenizer = in3120.SimpleTokenizer()
    print(repl_x_3.__doc__.strip())
    simple_repl("text", lambda t: [normalizer.normalize(token) for token in tokenizer.strings(normalizer.canonicalize(t))])


def repl_x_4():
    """
    Enter a wildcard query and locate matching MeSH terms.
    """
    print("Building permuterm index from MeSH corpus...")
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("mesh.txt")])
    expander = in3120.WildcardExpander((d["body"] for d in corpus))
    print(repl_x_4.__doc__.strip())
    simple_repl("text", lambda t: list(expander.expand(t)))


def repl_x_5():
    """
    Enter a document identifier and inspect its sparse document vector.
    """
    print("Indexing English news corpus...")
    analyzer = in3120.SimpleAnalyzer()
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("en.txt")])
    inverted_index = in3120.InMemoryInvertedIndex(corpus, ["body"], analyzer)
    stopwords0 = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("stopwords-en.txt")])
    stopwords = in3120.SimpleTrie.from_strings((d["body"] for d in stopwords0), analyzer)
    vectorizer = in3120.Vectorizer(corpus, inverted_index, stopwords)
    print(repl_x_5.__doc__.strip())
    print(f"There are {corpus.size()} documents.")
    simple_repl(f"[0, {corpus.size() - 1}]]", lambda i: {"document": corpus[int(i)], "vector": vectorizer.from_document(corpus[int(i)], ["body"]).top(100)})


def repl_x_6():
    """
    Enter some text and have it classified as a movie genre, using a Rocchio classifier.
    """
    print("Initializing Rocchio classifier from movie corpus...")
    analyzer = in3120.SimpleAnalyzer()
    movies = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("imdb.csv")])
    fields = ["title", "description"]
    inverted_index = in3120.DummyInMemoryInvertedIndex(movies, fields, analyzer)
    training_set = movies.split("genre", lambda v: v.split(","))
    stopwords0 = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("stopwords-en.txt")])
    stopwords = in3120.SimpleTrie.from_strings((d["body"] for d in stopwords0), analyzer)
    vectorizer = in3120.Vectorizer(movies, inverted_index, stopwords)
    classifier = in3120.RocchioClassifier(training_set, fields, vectorizer)
    print(repl_x_6.__doc__.strip())
    simple_repl("text", lambda t: list(classifier.classify(t)))


def repl_x_7():
    """
    Enter some text and detect its language, using a Rocchio classifier.
    """
    print("Initializing Rocchio classifier from news corpora...")
    analyzer = in3120.SimpleAnalyzer()
    languages = ["en", "no", "da", "de"]
    filenames = [data_path(f"{language}.txt") for language in languages]
    annotations = [{"lang": language} for language in languages]
    corpus = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), filenames, annotations)
    inverted_index = in3120.DummyInMemoryInvertedIndex(corpus, ["body"], analyzer)
    training_set = corpus.split("lang")
    stopwords0 = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("stopwords-en.txt")])
    stopwords = in3120.SimpleTrie.from_strings((d["body"] for d in stopwords0), analyzer)
    vectorizer = in3120.Vectorizer(corpus, inverted_index, stopwords)
    classifier = in3120.RocchioClassifier(training_set, ["body"], vectorizer)
    print(repl_x_7.__doc__.strip())
    print(f"Known languages are {languages}.")
    print("Returned scores are cosine similarities.")
    simple_repl("text", lambda t: list(classifier.classify(t)))


def repl_x_8():
    """
    Enter some query terms and locate a plausible result snippet in a chapter from Frankenstein.
    """
    print("Loading Frankenstein...")
    analyzer = in3120.Analyzer(in3120.PorterNormalizer(), in3120.SimpleTokenizer())
    finder = in3120.WindowFinder(analyzer)
    with open(data_path("frankenstein.txt"), "r", encoding="utf-8") as f:
        frankenstein = f.read()
    print(repl_x_8.__doc__.strip())
    def generate_snippet(query: str) -> str:
        result : in3120.WindowFinder.Result = finder.scan(frankenstein, query)
        if result is None:
            return "Unable to produce a snippet."
        if result.width > 75:
            return f"Suppressing snippet, window width is {result.width}."
        padding = 20
        before = frankenstein[max(0, result.begin - padding):result.begin]
        snippet = frankenstein[result.begin:result.end]
        after = frankenstein[result.end:min(len(frankenstein), result.end + padding)]
        return f"[...{before}]{snippet}[{after}...]"
    simple_repl("query", generate_snippet)


def repl_x_9():
    """
    Enter a movie genre and see its strongest classification features.
    """
    print("Loading movie corpus...")
    analyzer = in3120.SimpleAnalyzer()
    movies = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("imdb.csv")])
    fields = ["title", "description"]
    training_set = movies.split("genre", lambda v: v.split(","))
    stopwords0 = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), [data_path("stopwords-en.txt")])
    stopwords = in3120.SimpleTrie.from_strings((d["body"] for d in stopwords0), analyzer)
    selector = in3120.FeatureSelector(training_set, fields, analyzer)
    options = {"k": 10, "scoring": "mutual-information"}
    print(repl_x_9.__doc__.strip())
    print(f"Active options are {options}.")
    def select_features(query: str) -> Any:
        categories = [c.strip() for c in query.split(",")]
        match options["scoring"]:
            case "chi-square":
                return list(selector.chi_square(categories, options["k"], stopwords))
            case "mutual-information":
                return list(selector.mutual_information(categories, options["k"], stopwords))
            case _:
                raise ValueError(f"Unknown scoring method: {options['scoring']}")
    simple_repl("category", select_features)


def repl_x_10():
    """
    Enter some text and see what gets extracted based on an acronym-detection algorithm.
    """
    normalizer = in3120.SimpleNormalizer()
    extractor = in3120.AcronymExtractor()
    print(repl_x_10.__doc__.strip())
    simple_repl("text", lambda t: list(extractor.extract(normalizer.canonicalize(t))))


# The first letter of each key aligns with an obligatory assignment.
REPLS = {
    "a-1": repl_a_1,  # A.
    "a-2": repl_a_2,  # A.
    "b-1": repl_b_1,  # B-1.
    "b-2": repl_b_2,  # B-1.
    "b-3": repl_b_3,  # B-1.
    "b-4": repl_b_4,  # B-1.
    "b-5": repl_b_5,  # B-2.
    "c-1": repl_c_1,  # C-1.
    "c-2": repl_c_2,  # C-2.
    "d-1": repl_d_1,  # D-1.
    "d-2": repl_d_2,  # D-1.
    "d-3": repl_d_3,  # D-1.
    "d-4": repl_d_4,  # D-1.
    "d-5": repl_d_5,  # D-1.
    "e-1": repl_e_1,  # E-1.
    "e-2": repl_e_2,  # E-2.
    "e-3": repl_e_3,  # E-2.
    "x-1": repl_x_1,
    "x-2": repl_x_2,
    "x-3": repl_x_3,
    "x-4": repl_x_4,
    "x-5": repl_x_5,
    "x-6": repl_x_6,
    "x-7": repl_x_7,
    "x-8": repl_x_8,
    "x-9": repl_x_9,
    "x-10": repl_x_10,
}


def main():
    targets = sys.argv[1:]
    if not targets:
        print(f"{sys.argv[0]} [{'|'.join(REPLS.keys())}]")
        for k, f in REPLS.items():
            print(f"{Fore.LIGHTYELLOW_EX}{k}: {Fore.GREEN}{(f.__doc__ or '').strip()}{Style.RESET_ALL}")
    else:
        for target in targets:
            if target.lower() in REPLS:
                REPLS[target.lower()]()


if __name__ == "__main__":
    main()
