from dataclasses import dataclass, field
from typing import Dict, List

from generative_agents.simulation.maze import Tile


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
        if name in ('__getstate__', '__setstate__') or 'game_objects' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.game_objects, name)

    def __deepcopy__(self, memo):
        return ArenaMemory(game_objects=self.game_objects.copy())


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
        if name in ('__getstate__', '__setstate__') or 'arenas' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.arenas, name)

    def __deepcopy__(self, memo):
        return SectorMemory(arenas=self.arenas.copy())


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
        if name in ('__getstate__', '__setstate__') or 'sectors' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.sectors, name)

    def __deepcopy__(self, memo):
        return WorldMemory(sectors=self.sectors.copy())


@dataclass
class WorldMap:
    """
    The Agent's internal model of the world (Spatial Memory).
    """
    tree: Dict[str, List[WorldMemory]] = field(default_factory=dict)

    def add_tile(self, tile: Tile):
        """Adds a discovered tile to the map."""
        if not tile.world:
            return

        if tile.world not in self.tree:
            self.tree[tile.world] = WorldMemory()

        self.tree[tile.world].add(tile)
        
    def add(self, tile: Tile):
        """Legacy alias."""
        self.add_tile(tile)

    def __getitem__(self, key):
        return self.tree.get(key)
        
    def get_str_accessible_sectors(self, curr_world):
        if curr_world not in self.tree: return ""
        x = ", ".join(list(self.tree[curr_world].keys()))
        return x

    def get_str_accessible_sector_arenas(self, sector):
        if ":" not in sector: return ""
        curr_world, curr_sector = sector.split(":")
        if curr_world not in self.tree or curr_sector not in self.tree[curr_world]:
            return ""
        x = ", ".join(list(self.tree[curr_world][curr_sector].keys()))
        return x

    def get_str_accessible_arena_game_objects(self, arena):
        parts = arena.split(":")
        if len(parts) < 3: return ""
        curr_world, curr_sector, curr_arena = parts[0], parts[1], parts[2]
        
        try:
            x = ", ".join(
                list(self.tree[curr_world][curr_sector][curr_arena].game_objects.keys())
            )
        except Exception:
            return ""
        return x
        
    def __deepcopy__(self, memo):
        return WorldMap(tree=self.tree.copy())

    @property
    def known_addresses(self) -> set[str]:
        """Returns a set of all unique tile addresses the agent has physically perceived."""
        addresses = set()
        for world_name, world_memory in self.tree.items():
            addresses.add(world_name)
            if hasattr(world_memory, 'sectors'):
                for sector_name, sector_memory in world_memory.sectors.items():
                    addresses.add(f"{world_name}:{sector_name}")
                    if hasattr(sector_memory, 'arenas'):
                        for arena_name, arena_memory in sector_memory.arenas.items():
                            addresses.add(f"{world_name}:{sector_name}:{arena_name}")
                            if hasattr(arena_memory, 'game_objects'):
                                for object_name in arena_memory.game_objects.keys():
                                    addresses.add(f"{world_name}:{sector_name}:{arena_name}:{object_name}")
        return addresses

# Alias for backward compatibility during refactor
MemoryTree = WorldMap
