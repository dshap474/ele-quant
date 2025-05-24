# Elements of Quantitative Investing

This repository packages utilities and scripts used in the book *Elements of Quantitative Investing*.

The project is managed with [Poetry](https://python-poetry.org/) and follows a standard layout:

```
src/ele_quant/    # Python package with reusable utilities
scripts/          # Command line scripts
tests/            # Unit tests
```

To install the package along with its dependencies run:

```bash
poetry install
```

You can then run the `split_book.py` script to generate individual Markdown files from the `book` source file:

```bash
poetry run python scripts/split_book.py
```
