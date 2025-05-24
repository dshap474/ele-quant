#!/usr/bin/env python
"""Split the main `book` file into individual markdown files."""

from pathlib import Path
import re
from ele_quant import slugify


def main():
    book_path = Path("book")
    if not book_path.exists():
        raise FileNotFoundError("book file not found")

    lines = book_path.read_text().splitlines()

    indices = []
    titles = []
    for i, line in enumerate(lines):
        if line.startswith("**Introduction**"):
            indices.append(i)
            titles.append("Introduction")
        elif line.startswith("**Notation**"):
            indices.append(i)
            titles.append("Notation")
        elif re.match(r"\*\*(?:[Cc]hapter|CHAPTER)", line):
            match = re.match(r"\*\*(?:[Cc]hapter|CHAPTER)\s*(\d+)", line)
            if match:
                indices.append(i)
                titles.append(f"Chapter {match.group(1)}")
        elif line.startswith("**References**"):
            indices.append(i)
            titles.append("References")
        elif line.startswith("**Index**"):
            indices.append(i)
            titles.append("Index")

    indices.append(len(lines))

    output_dir = Path("markdown")
    output_dir.mkdir(exist_ok=True)

    for j in range(len(indices) - 1):
        start = indices[j]
        end = indices[j + 1]
        title = titles[j]
        filename = f"{j:02d}-{slugify(title)}.md"
        (output_dir / filename).write_text("\n".join(lines[start:end]))


if __name__ == "__main__":
    main()
