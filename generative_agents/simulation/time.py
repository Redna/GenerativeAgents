
import datetime
from enum import Enum

class DayType(Enum):
    FIRST_DAY = 1
    NEW_DAY = 2
    SAME_DAY = 3


class SimulationTime():
    """
    A class to represent the simulation time. It uses the the real time using datetime module but increments it every step with a 
    certain amount of time. 
    """

    def __init__(self, increment: int):
        self.increment = increment
        self.time = datetime.datetime.now()

    def tick(self):
        self.time += datetime
        datetime.timedelta(seconds=self.increment)

    def get(self):
        return self.time

    @property
    def today(self):
        return self.time.strftime("%A %B %d")
    
    @property
    def hour(self):
        """
        return the hour in format 01:00 AM
        """
        return self.time.strftime("%I:%M %p")
    
    @property
    def yesterday(self):
        return (self.time - datetime.timedelta(days=1)).strftime("%A %B %d")

    def as_string(self):
        """
        Returns the time as a string

        e.g. 30th of August 2021, 12:00:00
        """
        return self.time.strftime("%d %B %Y, %H:%M:%S")