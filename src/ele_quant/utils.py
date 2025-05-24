import re

__all__ = ["slugify"]

def slugify(text: str) -> str:
    """Convert a text string into a slug suitable for filenames.

    Parameters
    ----------
    text : str
        Input text to slugify.

    Returns
    -------
    str
        Slugified version of the text consisting of lowercase letters,
        numbers and hyphens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
