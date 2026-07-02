from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/panoptica#citation"

# Runtime toggle for the citation reminder (replaces the old env var approach).
_citation_reminder_enabled = True


def disable_citation_reminder() -> None:
    """Suppress the panoptica citation reminder for the rest of this process."""
    global _citation_reminder_enabled
    _citation_reminder_enabled = False


def enable_citation_reminder() -> None:
    """Re-enable the citation reminder (undoes :func:`disable_citation_reminder`)."""
    global _citation_reminder_enabled
    _citation_reminder_enabled = True


def citation_reminder(func):
    """Decorator to remind users to cite panoptica."""

    def wrapper(*args, **kwargs):
        global _citation_reminder_enabled
        if _citation_reminder_enabled:
            console = Console()
            console.rule("Thank you for using [bold]panoptica[/bold]")
            console.print(
                "Please support our development by citing",
                justify="center",
            )
            console.print(
                f"{CITATION_LINK} -- Thank you!",
                justify="center",
            )
            console.rule()
            console.line()
            _citation_reminder_enabled = False  # Show only once
        return func(*args, **kwargs)

    return wrapper
