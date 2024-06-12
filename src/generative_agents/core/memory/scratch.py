
from dataclasses import dataclass, field
import datetime
from typing import Tuple

from generative_agents.conversational.pipelines.identity import formulate_identity
from generative_agents.simulation.time import SimulationTime
from generative_agents.core.events import Action
from generative_agents.simulation.maze import Tile
from generative_agents.utils import hash_string
from generative_agents import global_state

@dataclass
class Scratch():
    name: str
    age: int
    tile: Tile
    home: Tile
    innate_traits: list[str]

    description: str = ""
    time: SimulationTime = None
    action: Action = None  
    learned_traits: list[str] = field(default_factory=list)
    vision_radius: int = 6
    attention_bandwith: int = 4
    retention: int = 5
    reflection_trigger_counter: int = 255
    reflection_trigger_max: int = 255
    planned_path: list[Tile] = field(default_factory=list)

    action_path_set = False
    lifestyle: str = "" # TODO generate lifestyle in reflect

    finished_action: list[Action] = field(default_factory=list)

    daily_requirements: str = ""
    current_status: str = ""
    daily_schedule: list[Tuple[str, int]] = None
    daily_schedule_hourly_organzied: list[Tuple[str, int]] = None
    hourly_activity_history: list[str] = field(default_factory=list)
    
    chatting_with: str = ""
    chatting_end_time: datetime.datetime = None
    chat: any = None
    # e.g., ["Dolores Murphy"] = self.vision_r
    chatting_with_buffer = dict()

    _identity: Tuple[str, str] = ("", "")
    _last_tick: int = -1

    def should_reflect(self):
        if (self.reflection_trigger_counter <= 0): 
            return True 
        return False
    
    def reset_reflection_counter(self):
        self.reflection_trigger_counter = self.reflection_trigger_max
        self.importance_ele_n = 0

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
            start = self.action.start_time
            if start.second != 0: 
                start = start.replace(second=0)
                start = (start + datetime.timedelta(minutes=1))
            end_time = (start + datetime.timedelta(minutes=self.action.duration))

        if end_time and self.time.time.strftime("%H:%M:%S") >= end_time.strftime("%H:%M:%S"): 
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
        today_min_elapsed += self.time.time.minute
        today_min_elapsed += advance

        x = 0
        try:
            for _, duration in self.daily_schedule_hourly_organzied: 
                x += duration
        except:
            print("ERROR")


        # We then calculate the current index based on that. 
        curr_index = 0
        elapsed = 0
        
        for _, duration in self.daily_schedule_hourly_organzied: 
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

    @property
    def identity(self):
        commonset = ""
        commonset += f"Name: {self.name}\n"
        commonset += f"Age: {self.age}\n"
        commonset += f"{self.description}"
        commonset += f"Innate traits: {self.innate_traits}\n"
        commonset += f"Learned traits: {self.learned_traits}\n"

        commonset += f"Lifestyle: {self.lifestyle}\n"
        commonset += f"Daily plan requirement: {self.daily_requirements}\n"

        key, cached_identity = self._identity
        
        if key != hash_string(commonset) and global_state.tick != self._last_tick:
            new_hash = hash_string(commonset)
            
            if self.action:
                commonset += f"Currently: {self.action.event.description}\n"
            commonset += f"Current Date: {self.time.today}\n"

            cached_identity = formulate_identity(self.name, commonset)
            self.description = cached_identity
            self._identity = (new_hash, cached_identity)
            self._last_tick = global_state.tick

        return cached_identity