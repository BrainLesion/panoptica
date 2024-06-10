import time


def measure_time(func):
    """Decorator to measure the time it takes to execute a function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if hasattr(kwargs, "verbose") and getattr(kwargs, "verbose"):
            print(f"-- {func.__name__} took {elapsed_time} seconds to execute.")
        return result

    return wrapper
