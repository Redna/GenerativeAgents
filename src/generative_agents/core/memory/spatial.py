from dataclasses import dataclass, field
from typing import Dict, List, Set
from pydantic import BaseModel

from generative_agents.simulation.maze import Maze, Tile

@dataclass
class ArenaMemory:
    game_objects: Dict[str, Tile] = field(default_factory=dict)

    def add(self, tile: Tile):
        if not tile.game_object:
            return

        self.game_objects[tile.game_object] = tile
    
    def __getitem__(self, key):
        return self.game_objects[key]

    def __getattr__(self, name):
        return getattr(self.game_objects, name)


@dataclass
class SectorMemory:
    arenas: Dict[str, List[ArenaMemory]] = field(default_factory=dict)

    def add(self, tile: Tile):
        if not tile.arena:
            return

        if tile.arena not in self.arenas:
            self.arenas[tile.arena] = ArenaMemory()

        self.arenas[tile.arena].add(tile)

    def __getitem__(self, key):
        return self.arenas.get(key)

    def __setitem__(self, key, value):
        self.arenas[key] = value

    def __getattr__(self, name):
        return getattr(self.arenas, name)

@dataclass
class WorldMemory:
    sectors: Dict[str, List[SectorMemory]] = field(default_factory=dict)

    def add(self, tile: Tile):
        if not tile.sector:
            return

        if tile.sector not in self.sectors:
            self.sectors[tile.sector] = SectorMemory()

        self.sectors[tile.sector].add(tile)

    def __getitem__(self, key):
        return self.sectors.get(key)

    def __setitem__(self, key, value):
        self.sectors[key] = value

    def __getattr__(self, name):
        return getattr(self.sectors, name)

@dataclass
class MemoryTree:
    tree: Dict[str, List[WorldMemory]] = field(default_factory=dict)
                    
    def add(self, tile: Tile):
        if not tile.world:
            return

        if tile.world not in self.tree:
            self.tree[tile.world] = WorldMemory()

        self.tree[tile.world].add(tile)
    
    def __getitem__(self, key):
        return self.tree.get(key)

    def __setitem__(self, key, value):
        self.tree[key] = value

    def __getattr__(self, name):
        return getattr(self.tree, name)

    def get_str_accessible_sectors(self, curr_world):
        """
        Returns a summary string of all the arenas that the persona can access 
        within the current sector. 

        Note that there are places a given persona cannot enter. This information
        is provided in the persona sheet. We account for this in this function. 

        INPUT
        None
        OUTPUT 
        A summary string of all the arenas that the persona can access. 
        EXAMPLE STR OUTPUT
        "bedroom, kitchen, dining room, office, bathroom"
        """
        x = ", ".join(list(self.tree[curr_world].keys()))
        return x

    def get_str_accessible_sector_arenas(self, sector):
        """
        Returns a summary string of all the arenas that the persona can access 
        within the current sector. 

        Note that there are places a given persona cannot enter. This information
        is provided in the persona sheet. We account for this in this function. 

        INPUT
            None
        OUTPUT 
            A summary string of all the arenas that the persona can access. 
        EXAMPLE STR OUTPUT
            "bedroom, kitchen, dining room, office, bathroom"
        """
        curr_world, curr_sector = sector.split(":")
        if not curr_sector:
            return ""
        x = ", ".join(list(self.tree[curr_world][curr_sector].keys()))
        return x

    def get_str_accessible_arena_game_objects(self, arena):
        """
        Get a str list of all accessible game object_s that are in the arena. If 
        temp_address is specified, we return the object_s that are available in
        that arena, and if not, we return the object_s that are in the arena our
        persona is currently in. 

        INPUT
            temp_address: optional arena address
        OUTPUT 
            str list of all accessible game object_s in the gmae arena. 
        EXAMPLE STR OUTPUT
            "phone, charger, bed, nightstand"
        """
        curr_world, curr_sector, curr_arena = arena.split(":")

        if not curr_arena:
            return ""

        try:
            x = ", ".join(list(self.tree[curr_world][curr_sector][curr_arena].game_objects.keys()))
        except:
            return None
        return x