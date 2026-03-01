# app/extensions/bibliography.py

from pathlib import Path
import tempfile
import os
import pypandoc

BASE_DIR = Path(__file__).resolve().parent
CSL_DIR = BASE_DIR / "csl"

CSL_FILES = {
    "apa": CSL_DIR /  "apa.csl",
    "ieee": CSL_DIR / "ieee.csl",
    "mla": CSL_DIR / "modern-language-association.csl",
}


def bibtex_to_multiple_styles(bibtex_string: str) -> dict:
    """
    Convert a BibTeX string into APA, IEEE, and MLA
    formatted bibliography strings.
    """

    results = {}

    # Create temporary .bib file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as bib_file:
        bib_file.write(bibtex_string)
        bib_path = bib_file.name

    try:
        # Convert path to forward slashes for cross-platform compatibility
        # Pandoc/YAML expects forward slashes even on Windows
        bib_path_normalized = bib_path.replace('\\', '/')
        
        markdown = f"""
---
bibliography: {bib_path_normalized}
nocite: '@*'
---
"""

        for style, csl_path in CSL_FILES.items():

            output = pypandoc.convert_text(
                markdown,
                to="plain",
                format="markdown",
                extra_args=[
                    "--citeproc",
                    f"--csl={str(csl_path)}"
                ],
            )

            results[style] = output.strip()

    finally:
        os.remove(bib_path)

    return results
