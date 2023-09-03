
from dataclasses import dataclass
import datetime
from typing import List, Tuple
from queue import Queue
from simulation.maze import Tile
from simulation.time import SimulationTime
from core.events import Action

@dataclass
class Scratch():
    vision_radius: int = 4
    attention_bandwith: int = 3
    retention: int = 5
    planned_path: List[Tile] = []
    tile: Tile = None
    home: Tile = None
    time: SimulationTime = None
    lifestyle: str = ""
    identity: str = ""
    action: Action = Queue()
    action_queue: Queue = None
    daily_requirements: str = ""
    current_status: str = ""
    daily_schedule: List[Tuple[str, int]] = None
    daily_schedule_hourly_organzied: List[Tuple[str, int]] = None
    hourly_activity_history: List[str] = []
    
    chatting_with: str = ""
    chatting_end_time: datetime.datetime = None
    # e.g., ["Dolores Murphy"] = self.vision_r
    chatting_with_buffer = dict()

    def is_action_finished(self):
        """
        Checks whether the self.Action instance has finished.  

        INPUT
        curr_datetime: Current time. If current time is later than the action's
                        start time + its duration, then the action has finished. 
        OUTPUT 
        Boolean [True]: Action has finished.
        Boolean [False]: Action has not finished and is still ongoing.
        """
        if not self.action: 
            return True
        
        if self.chatting_with: 
            end_time = self.chatting_end_time
        else: 
            start = self.act_start_time
            if start.second != 0: 
                start = start.replace(second=0)
                start = (start + datetime.timedelta(minutes=1))
        end_time = (start + datetime.timedelta(minutes=self.act_duration))

        if end_time.strftime("%H:%M:%S") == self.curr_time.strftime("%H:%M:%S"): 
            return True
        return False

    def get_daily_schedule_index(self, advance=0):
        """
        We get the current index of self.daily_schedule. 

        Recall that self.f_daily_schedule stores the decomposed action sequences 
        up until now, and the hourly sequences of the future action for the rest
        of today. Given that self.f_daily_schedule is a list of list where the 
        inner list is composed of [task, duration], we continue to add up the 
        duration until we reach "if elapsed > today_min_elapsed" condition. The
        index where we stop is the index we will return. 

        INPUT
        advance: Integer value of the number minutes we want to look into the 
                future. This allows us to get the index of a future timeframe.
        OUTPUT 
        an integer value for the current index of f_daily_schedule.
        """
        # We first calculate teh number of minutes elapsed today. 
        today_min_elapsed = 0
        today_min_elapsed += self.time.time.hour * 60
        today_min_elapsed += self.time.minute
        today_min_elapsed += advance

        x = 0
        for _, duration in self.daily_schedule: 
            x += duration
            x = 0
            
        for _, duration in self.daily_schedule_hourly_organzied: 
            x += duration

        # We then calculate the current index based on that. 
        curr_index = 0
        elapsed = 0
        
        for _, duration in self.f_daily_schedule: 
            elapsed += duration
            if elapsed > today_min_elapsed: 
                return curr_index
            curr_index += 1

        return curr_index
    
    def random_path(self, maze):
        while not self.scratch.planned_path:
            print("Searching for a suitable path for the agent")
            target_tile = maze.get_random_tile(self.tile)
            print("Setting new target to ", target_tile)
            path = maze.find_path(self.tile, target_tile)
            if not path:
                print("Here is no path to the target, trying again")
            else:
                print("Found a path to the target")
                return path

