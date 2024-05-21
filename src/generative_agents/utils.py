
from datetime import datetime
import hashlib

from pathlib import Path


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