from colored import fg, attr

def log_info(message: str) -> None:
    print(fg("cyan") + "[INFO] " + attr("reset") + message)


def log_warning(message: str) -> None:
    print(fg("yellow") + "[WARNING] " + attr("reset") + message)

def log_error(message: str) -> None:
    print(fg("red") + "[ERROR] " + attr("reset") + message)

def log_debug(message: str) -> None:
    print(fg("blue") + "[DEBUG] " + attr("reset") + message)

def log_success(message: str) -> None:
    print(fg("green") + "[SUCCESS] " + attr("reset") + message)

def log_critical(message: str) -> None:
    print(fg("magenta") + "[CRITICAL] " + attr("reset") + message)

def log_custom(level: str, message: str, color: str) -> None:
    print(fg(color) + f"[{level}] " + attr("reset") + message)