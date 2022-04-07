""" CLI entrypoint for the scBoolSeq tool.

scBoolSeq:
scRNA-Seq data binarisation and synthetic generation from Boolean dynamics.

author: "Gustavo Magaña López"
credits: "BNediction ; Institut Curie"
"""

from ._cli import scBoolSeqCLIParser


def main():
    """Call the scBoolSeq automated runner"""
    scBoolSeqCLIParser()


if __name__ == "__main__":
    main()
