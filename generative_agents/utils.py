
from datetime import datetime


def get_date_string(dt: datetime) -> str:
    """ Returns a string representation of the time. """
    return dt.strftime("%A %B %d, %Y")

def get_time_string(dt: datetime) -> str:
    return dt.strftime("%I:%M%p.")