
import datetime
from enum import Enum

from generative_agents import global_state

class DayType(Enum):
    FIRST_DAY = 1
    NEW_DAY = 2
    SAME_DAY = 3


class SimulationTime():
    """
    A class to represent the simulation time. It uses the the real time using datetime module but increments it every step with a 
    certain amount of time. 
    """

    def __init__(self, increment: int, from_time_string: str = None):
        self.increment = increment
        self.time = datetime.datetime.today()

        if from_time_string:
            date_string = self.time.strftime('%Y-%m-%d')
            datetime_string = f"{date_string} {from_time_string}"
            self.time = datetime.datetime.strptime(datetime_string, '%Y-%m-%d %H:%M')



    def tick(self):
        self.time += datetime.timedelta(seconds=self.increment)
        global_state.tick += 1

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