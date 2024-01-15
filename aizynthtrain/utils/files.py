"""Module containing various file utilities"""


def prefix_filename(prefix: str, postfix: str) -> str:
    """
    Construct pre- and post-fixed filename

    :param prefix: the prefix, can be empty
    :param postfix: the postfix
    :return: the concatenated string
    """
    if prefix:
        return prefix + "_" + postfix
    return postfix
