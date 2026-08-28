# Introduction

This is the companion GitHub repository for the [IN3120](https://www.uio.no/studier/emner/matnat/ifi/IN3120/)/[IN4120](https://www.uio.no/studier/emner/matnat/ifi/IN4120/) course, used to disseminate obligatory assignments and other relevant materials.

You will not be pushing anything to this repository, you will only be [pulling from it](https://github.com/git-guides/git-pull). First, you will need to [clone it](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository).

There are five obligatory assignments in IN3120 and IN4120. In addition, there will be an obligatory [science fair](./science-fair.md) for the IN4120 students.

## Code

The coding assignments assume basic familiarity with the Python language. At least [version 3.10](https://www.python.org/downloads/) will be assumed, version 3.12.9 is known to work fine with the current project dependencies. You can use whatever development environment you want, but you will probably be more productive and have an easier time if you use a good IDE. We recommend [Visual Studio Code](https://code.visualstudio.com/). [PyCharm](https://www.jetbrains.com/pycharm/) is another great choice. You can use another set of tools if you want, but then don't expect help with solving challenges related to setup or tooling.

You will be provided with some "precode" or "starter code", i.e., a set of helper classes and functions that you can make use of so that you don't have to start the coding assignments completely from scratch. This precode also sets some structure on how you implement the assignments. Please familiarize yourself with what's available. The precode is commented and has some illustrative usage examples. Note the following three folders in the repository:

* [`in3120`](../in3120/): The actual precode. See [here](./guide.md) for a short walkthrough guide.
* [`tests`](../tests/): Tests and REPL-related code. To familiarize yourself with the precode, this might be a good place to start after having read the walkthrough guide.
* [`data`](../data/): Various corpora used by the tests.

The precode is in many places annotated with [type hints](https://www.python.org/dev/peps/pep-0484/). This doesn't make Python a statically typed language, but are just hints that are possible to ignore and abuse. They do convey intent and serve as a kind of additional documentation, though, and enable Visual Studio Code and other IDEs to give you better and richer programming support.

Please strive to create readable and modular code. If the [linter](https://en.wikipedia.org/wiki/Lint_(software)) in your IDE warns you about problems, please address them.

## Preliminaries

We recommend that you first set up a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html). This will help you avoid some potential packaging conflicts. In the following, we'll assume that you have created and activated such a virtual environment.

This repository includes few module dependencies, since the main purpose of this repository is pedagogical. In a select few cases, though, we import a few non-standard packages when the purpose is to demonstrate their use and the concepts they expose rather than to demonstrate how they are implemented. To install these dependencies, invoke the commands listed below. You only need to do this once.

```text
>python3 -m pip install -r requirements.txt
>python3 -m spacy download en_core_web_md
```

Note that, depending on your platform and Python installation, the Python interpreter might not be invoked via `python3` but rather via `python` or simply `py`.

Alternatively, dependencies can be installed through Anaconda. First, [install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions). Then, create and activate a new virtual environment as follows:

```text
>conda create -n in3120 python=3.11.9 numpy=1.26.4 conda-forge::spacy=3.7.5 pytorch::faiss-cpu=1.10.0 colorama=0.4.6
>conda activate in3120
```

The version numbers listed as arguments to `conda create` should match those listed in [`requirements.txt`](../requirements.txt).

## Assignments

Individual assignments are linked to below.

Please read the assignment texts carefully before you begin. Answers to common questions are surprisingly often answered in the assignment texts. This is worth repeating: Please read the assignment texts carefully before you begin. Please also make sure that you understand the precode and the unit tests that your code is expected to pass.

* [Assignment A](./assignment-a.md).
* [Assignment B-1](./assignment-b-1.md) or [Assignment B-2](./assignment-b-2.md).
* [Assignment C-1](./assignment-c-1.md) or [Assignment C-2](./assignment-c-2.md).
* [Assignment D-1](./assignment-d-1.md) or [Assignment D-2](./assignment-d-2.md).
* [Assignment E-1](./assignment-e-1.md) or [Assignment E-2](./assignment-e-2.md).

You can choose between doing either Assignment B-1 _or_ Assignment B-2. You do not have to do them both. Similarly, you can choose between doing either Assignment C-1 _or_ Assignment C-2, Assignment D-1 _or_ Assignment D-2, and Assignment E-1 _or_ Assignment E-2.

Lastly, please read the assignment texts carefully before you begin.

## Tests

Common for the precode is that a `NotImplementedError` exception is raised in places where you are meant to provide a working implementation. After having provided working implementations, all tests should pass. Specifically, the following should work and run without errors:

```text
>cd tests
>python3 assignments.py
```

The above invocation of [`assignments.py`](../tests/assignments.py) runs all tests for all assignments. If you want to only run the tests for, say, assignments A and C-2, you can pass this as command line arguments:

```text
>cd tests
>python3 assignments.py a c-2
```

An alternate way of running all tests is shown below. See [documentation](https://docs.python.org/3/library/unittest.html#command-line-interface) for the `unittest` module for more information.

```text
>cd tests
>python3 -m unittest
```

Making tests pass for one assignment should not result in tests breaking for previous assignments. You should think of passing tests as a necessary but not sufficient condition for having your solution accepted, e.g., an implementation that makes tests pass through emitting hardcoded output targeting the tests will make the tests pass but is obviously not an acceptable solution.

If your code raises no exceptions and passes all the `assert` statements, you should see `OK` printed to the console at the end of the test run. E.g.:

```text
...
----------------------------------------------------------------------
Ran 329 tests in 21.828s

OK
```

If you are using PyCharm as your IDE, tests can also be run and debugged from within PyCharm: Open up the file containing the tests in PyCharm and next to the test class or test method, in the left margin of your editor, look for a green arrow as shown [here](https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html). Clicking it will bring up a context-sensitive menu. Using the default setting of `unittest` as the default test runner should work fine.

Similarly, if you are using Visual Studio Code as your IDE, tests can be run from within the IDE. Your configuration should look something like this:

```json
>more .\.vscode\settings.json
{
    "python.testing.unittestArgs": [
        "-v",
        "-s",
        "${workspaceFolder}/tests",
        "-p",
        "test_*.py"
    ],
    "python.testing.cwd": "${workspaceFolder}/tests",
    "python.testing.pytestEnabled": false,
    "python.testing.unittestEnabled": true
}
```

Simple [REPLs](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) for interactive ad hoc testing from the command line can be started by specifying alternate targets, e.g.:

```text
>cd tests
>python3 repl.py c-1
```

If you do not specify any targets to [`repl.py`](../tests/repl.py) you will be given a menu of the targets available.

## Solutions

The different assignments have different deadlines. Please see the assignment texts for exact dates.

Your implementations should make all tests pass. Please read the assignment texts carefully.

All your implementations should be reasonably efficient, complexity-wise. E.g., if the assignment specifies that your solution should run in logarithmic time, a solution that yields the correct output and passes all tests but that runs in quadratic time won't pass muster.

When turning in your solutions, please note the following delivery constraints that enable partial automation on the receiver side and that make life easier for the teaching assistants:

* Submit your solution using [Devilry](https://devilry.ifi.uio.no/). No emails with attachments, please.
* Your implementation should only make use of [standard Python libraries](https://docs.python.org/3/library/index.html), unless explicitly noted otherwise. No imports of non-standard libraries, please, unless otherwise noted.
* Your implementation should be fully contained in the supplied `*.py` files. Don't introduce any new files, please.
* The files you submit for your implementation should only be the `*.py` files where there are missing implementations per the assignment. Don't submit untouched files from this repository, please, and don't spread your implementation across other files.
* Your implementation should adhere to the APIs of the abstract base classes, without modifications.
* If you want to submit a single-file archive, you are limited to `*.zip`, `*.tar` and `*.tar.gz` formats. No other archive formats, please.

Solutions are made progressively available as the course unfolds. Simply do a `git pull` to get the solutions when published. If you get merge conflicts you are unable to resolve, as a last resort you can always nuke your copy of this repository and then re-clone it.

Some assignments have dependencies in the sense of one assignment relying on code developed in an earlier assignment. As solutions are published, it is strongly recommended that you use the published solutions going forwards.

## Other

In addition to code for the programming assignments, this repository also contains supplementary materials for this course. In particular, note the following folders:

* [`papers`](../papers/): Various papers that supplement the textbook and that are used in this course.
* [`slides`](../slides/): Slides that are used as part of the lectures.
* [`exams`](../exams/): Final exams from previous years, including rough solution sketches.

Lastly, some notes are available regarding [administrative information](./administrivia.md) about the course, plus some [background context](./notes.md) about the course.
