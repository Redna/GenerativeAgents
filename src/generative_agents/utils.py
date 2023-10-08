
from datetime import datetime

from pathlib import Path


def get_date_string(dt: datetime) -> str:
    """ Returns a string representation of the time. """
    return dt.strftime("%A %B %d, %Y")

def get_time_string(dt: datetime) -> str:
    return dt.strftime("%I:%M%p.")

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent