import os


def darknet(term):
    """[a function to invoke the darknet command]

    Args:
        term ([string]): [the command that needs to be executed after the darknet invocation]
    """
    command = f"darknet {term}"

    os.system(command)
