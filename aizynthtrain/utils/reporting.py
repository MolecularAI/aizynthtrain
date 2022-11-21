"""Module containing routines to create reports"""
import tempfile
import argparse
from typing import Optional, Sequence, Any

import nbconvert
import nbformat
import jupytext
import papermill


def create_html_report_from_notebook(
    notebook_path: str, html_path: str, python_kernel: str, **parameters: Any
) -> None:
    """
    Execute a Jupyter notebook and create an HTML file from the output.

    :param notebook_path: the path to the Jupyter notebook in py-percent format
    :param html_path: the path to the HTML output
    :param python_kernel: the python kernel to execute the notebook in
    :param parameters: additional parameters given to the notebook
    """
    _, input_notebook = tempfile.mkstemp(suffix=".ipynb")
    _, output_notebook = tempfile.mkstemp(suffix=".ipynb")

    notebook = jupytext.read(notebook_path, fmt="py:percent")
    jupytext.write(notebook, input_notebook, fmt="ipynb")

    papermill.execute_notebook(
        input_notebook,
        output_notebook,
        kernel_name=python_kernel,
        language="python",
        parameters=parameters,
    )

    with open(output_notebook, "r") as fileobj:
        notebook_nb = nbformat.read(fileobj, as_version=4)
    exporter = nbconvert.HTMLExporter()
    exporter.exclude_input = True
    notebook_html, _ = exporter.from_notebook_node(notebook_nb)
    with open(html_path, "w") as fileobj:
        fileobj.write(notebook_html)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface for report generation"""
    parser = argparse.ArgumentParser("Run a Jupyter notebook to produce a report")
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--python_kernel", required=True)
    args, extra_args = parser.parse_known_args(args=args)

    keys = [arg[2:] for arg in extra_args[:-1:2]]
    kwargs = dict(zip(keys, extra_args[1::2]))

    create_html_report_from_notebook(
        args.notebook, args.report_path, args.python_kernel, **kwargs
    )


if __name__ == "__main__":
    main()
