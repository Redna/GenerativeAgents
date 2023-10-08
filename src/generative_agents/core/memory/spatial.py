from typing import Dict, List
from pydantic import BaseModel

from generative_agents.simulation.maze import Tile

class ArenaMemory(BaseModel):
    game_objects: List[str] = []

    def add(self, tile: Tile):
        if not tile.game_object:
            return

        self.game_objects += [tile.game_object]


class SectorMemory(BaseModel):
    arenas: Dict[str, List[ArenaMemory]] = {}

    def add(self, tile: Tile):
        if not tile.arena:
            return

        if tile.arena not in self.arenas:
            self.arenas[tile.arena] = ArenaMemory()

        self.arenas[tile.arena].add(tile)


class WorldMemory(BaseModel):
    sectors: Dict[str, List[SectorMemory]] = {}

    def add(self, tile: Tile):
        if not tile.sector:
            return

        if tile.sector not in self.sectors:
            self.sectors[tile.sector] = SectorMemory()

        self.sectors[tile.sector].add(tile)


class MemoryTree(BaseModel):
    tree: Dict[str, List[WorldMemory]] = {}

    def add(self, tile: Tile):
        if not tile.world:
            return

        if tile.world not in self.tree:
            self.tree[tile.world] = WorldMemory()

        self.tree[tile.world].add(tile)

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

    def get_str_accessible_arena_game_object_s(self, arena):
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
            x = ", ".join(list(self.tree[curr_world][curr_sector][curr_arena]))
        except:
            x = ", ".join(
                list(self.tree[curr_world][curr_sector][curr_arena.lower()]))
        return x