import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os

    # Walk up one level from notebooks/ to reach the project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return


app._unparsable_cell(
    r"""
    import marimo as mo
    from 
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
