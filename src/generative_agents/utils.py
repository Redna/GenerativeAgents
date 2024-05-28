
from contextlib import contextmanager
from datetime import datetime
import hashlib

from pathlib import Path
from time import perf_counter, time

from functools import wraps
from colorama import Fore, Style, Back

@contextmanager
def colored(style, fore, back):
    print(style + fore + back, end="")
    yield
    print(Style.RESET_ALL, end="")


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        with colored(Style.BRIGHT, Fore.CYAN, Back.BLACK):
            print(f"{func.__qualname__} took {end - start}")
        return result
    return wrapper
    



def get_date_string(dt: datetime) -> str:
    """ Returns a string representation of the time. """
    return dt.strftime("%A %B %d, %Y")

def get_time_string(dt: datetime) -> str:
    return dt.strftime("%I:%M %p")

def time_string_to_time(time: str) -> datetime:
    return datetime.strptime(time, "%I:%M %p")

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def hash_string(s: str) -> int:
    """ Returns a hash of the string. """
    return hashlib.sha1(s.encode()).hexdigest()

def hour_string_to_time(hour: str):
    return datetime.strptime(str(hour), "%H")