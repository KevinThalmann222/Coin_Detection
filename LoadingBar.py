def progress_bar(progresses: int, total: int, length: int = 50, text: str = "") -> None:
    """create a loadingbar

    Args:
        progresses (int): num of current step
        total (int): num of total steps
        text (str, optional): loading status. Defaults to "".
    """
    percent = length * ((progresses + 1) / float(total))
    bar = "=" * int(percent) + "-" * (length - int(percent))
    print(f"\r[{bar}] {100/length*percent:.2f}% {text}", end="\r")
    if percent == length:
        print("\r")
