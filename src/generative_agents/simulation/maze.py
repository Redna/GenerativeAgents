import csv
from enum import Enum
from functools import lru_cache
import heapq
import random
from typing import List, Tuple
from pathfinding.core.grid import Grid, GridNode
from pathfinding.finder.a_star import AStarFinder

   # create a tile class holding the following structure 
   # {'world': 'double studio', 
    #         'sector': 'double studio', 'arena': 'bedroom 2', 
    #         'game_object': 'bed', 'spawning_location': 'bedroom-2-a', 
    #         'collision': False,
    #         'events': {('double studio:double studio:bedroom 2:bed',
    #                    None, None)}} 

# Set current workdir to file location
import os

from generative_agents.utils import get_project_root

class Level(Enum):
    WORLD = "world"
    SECTOR = "sector"
    ARENA = "arena"
    GAME_OBJECT = "game_object"
    SPAWNING_LOCATION = "spawning_location"

class Tile:
    def __init__(self, x,y, world, sector, arena, game_object, spawning_location, collision, events):
        super().__init__()
        self.x = x
        self.y = y
        self.world = world
        self.sector = sector
        self.arena = arena
        self.game_object = game_object
        self.spawning_location = spawning_location
        self.collision = collision
        self.events = events
    
    def get_unique_name(self):
        address = ""

        if self.world: 
            address += self.world
        if self.sector:
            address += f":{self.sector}"
        if self.arena:
            address += f":{self.arena}"
        if self.game_object:
            address += f":{self.game_object}"
        if self.spawning_location:
            address = f'<spawn_loc>{self.spawning_location}'

        return address
    
    def get_path(self, level: Level):
        path = f"{self.world}"

        if level == level.WORLD: 
            return path
        else: 
            path += f":{self.sector}"
        
        if level == level.SECTOR: 
            return path
        else: 
            path += f":{self.arena}"

        if level == level.ARENA: 
            return path
        else: 
            path += f":{self.game_object}"

        return path
        
    def is_sector(self):
        return self.sector != ""
    
    def is_arena(self):
        return self.arena != ""
    
    def is_game_object(self):
        return self.game_object != ""
    
    def is_spawning_location(self):
        return self.spawning_location != ""
    
    def is_walkable(self):
        return not self.collision

    def __str__(self):
        return f"Tile(world={self.world}, sector={self.sector}, arena={self.arena}, game_object={self.game_object}, spawning_location={self.spawning_location}, collision={self.collision}, events={self.events})"

    def __repr__(self):
        return self.__str__()
    
    def __gt__(self, other: 'Tile') -> bool:
        return (self.x, self.y) > (other.x, other.y)
    
    def __lt__(self, other: 'Tile') -> bool:
        return (self.x, self.y) < (other.x, other.y)
    
    def __eq__(self, other: 'Tile') -> bool:
        return (self.x, self.y) == (other.x, other.y)
    
    def __hash__(self) -> int:
        return hash((self.get_unique_name, self.x, self.y))

class SimplePathFinder():
    def __init__(self, grid: List[List[Tile]]):
        self.grid = grid
    
    def find_path(self, start, end):
        open_set = []
        closed_set = set()
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {pos: float('inf') for row in self.grid for pos in row}
        g_score[start] = 0
        f_score = {pos: float('inf') for row in self.grid for pos in row}
        f_score[start] = self._heuristic(start, end)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end:
                path = self._reconstruct_path(came_from, current)
                return path

            closed_set.add(current)

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, end)
                    if neighbor not in [pos for _, pos in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    @staticmethod
    def _heuristic(start, end):
        return abs(start.x - end.x) + abs(start.y - start.y)

    def _get_neighbors(self, pos):
        neighbors = []
        x = pos.x
        y = pos.y

        if x > 0 and self.grid[y][x - 1].is_walkable():
            neighbors.append(self.grid[y][x - 1])
        if x < len(self.grid[0]) - 1 and self.grid[y][x + 1].is_walkable():
            neighbors.append(self.grid[y][x + 1])
        if y > 0 and self.grid[y - 1][x].is_walkable():
            neighbors.append(self.grid[y - 1][x])
        if y < len(self.grid) - 1 and self.grid[y + 1][x].is_walkable():
            neighbors.append(self.grid[y + 1][x])
        return neighbors

    @staticmethod
    def _reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


class Maze:

    maze_name = "The Ville"
    maze_width = 140
    maze_height = 100
    tile_size = 32

    def __init__(self):
        self.finder = AStarFinder()
        self.maze = []

        # READING IN SPECIAL BLOCKS
        # Special blocks are those that are colored in the Tiled map. 

        # Here is an example row for the arena block file: 
        # e.g., "25335, Double Studio, Studio, Common Room"
        # And here is another example row for the game object block file: 
        # e.g, "25331, Double Studio, Studio, Bedroom 2, Painting"

        # Notice that the first element here is the color marker digit from the 
        # Tiled export. Then we basically have the block path: 
        # World, Sector, Arena, Game Object -- again, these paths need to be 
        # unique within an instance of Reverie. 
        blocks_folder = os.path.join(get_project_root(), "assets/matrix/special_blocks")

        world_blocks = self.read_special_blocks(blocks_folder + "/world_blocks.csv")
        world_block = world_blocks[0][-1]

        sector_blocks = self.read_special_blocks(blocks_folder + "/sector_blocks.csv")
        sector_blocks_dict = {block[0]: block[-1] for block in sector_blocks}
        
        arena_blocks = self.read_special_blocks(blocks_folder + "/arena_blocks.csv")
        arena_blocks_dict = {block[0]: block[-1] for block in arena_blocks}

        game_object_blocks = self.read_special_blocks(blocks_folder + "/game_object_blocks.csv")
        game_object_blocks_dict = {block[0]: block[-1] for block in game_object_blocks}

        spawning_location_blocks = self.read_special_blocks(blocks_folder + "/spawning_location_blocks.csv")
        spawning_location_blocks_dict = {block[0]: block[-1] for block in spawning_location_blocks}

        # [SECTION 3] Reading in the matrices 
        # This is your typical two dimensional matrices. It's made up of 0s and 
        # the number that represents the color block from the blocks folder. 
        maze_folder = os.path.join(get_project_root(), "assets/matrix/maze")

        collision_maze_raw = self.read_special_blocks(maze_folder + "/collision_maze.csv")[0]
        sector_maze_raw = self.read_special_blocks(maze_folder + "/sector_maze.csv")[0]
        arena_maze_raw = self.read_special_blocks(maze_folder + "/arena_maze.csv")[0]
        game_object_maze_raw = self.read_special_blocks(maze_folder + "/game_object_maze.csv")[0]
        spawning_location_maze_raw = self.read_special_blocks(maze_folder + "/spawning_location_maze.csv")[0]

        # [SECTION 4] Converting the matrices to 2d lists
        # We need to convert the matrices to 2d lists.
        # We also need to convert the 0s to None.

        collision_maze = self.convert_flat_list_to_2d_list(collision_maze_raw, self.maze_width)
        sector_maze = self.convert_flat_list_to_2d_list(sector_maze_raw, self.maze_width)
        arena_maze = self.convert_flat_list_to_2d_list(arena_maze_raw, self.maze_width)
        game_object_maze = self.convert_flat_list_to_2d_list(game_object_maze_raw, self.maze_width)
        spawning_location_maze = self.convert_flat_list_to_2d_list(spawning_location_maze_raw, self.maze_width)

        # [SECTION 5] Creating the maze
        # We need to create the maze.
        
        self.tiles = []
        
        self.grid = Grid(self.maze_width, self.maze_height)

        for i in range(self.maze_height):
            row = []
            for j in range(self.maze_width):
                sector = sector_blocks_dict[sector_maze[i][j]] if sector_maze[i][j] in sector_blocks_dict else ""
                arena = arena_blocks_dict[arena_maze[i][j]] if arena_maze[i][j] in arena_blocks_dict else ""
                game_object = game_object_blocks_dict[game_object_maze[i][j]] if game_object_maze[i][j] in game_object_blocks_dict else ""
                spawning_location = spawning_location_blocks_dict[spawning_location_maze[i][j]] if spawning_location_maze[i][j] in spawning_location_blocks_dict else ""
                collision = collision_maze[i][j] != "0"
                row += [Tile(j, i, world_block, sector, arena, game_object, spawning_location, collision, dict())]           
                node = self.grid.node(j,i)
                node.walkable = not collision
                node.weight = 0 if collision else 1

            self.tiles += [row]

        # Reverse tile access. 
        # <self.address_tiles> -- given a string address, we return a set of all 
        # tile coordinates belonging to that address (this is opposite of  
        # self.tiles that give you the string address given a coordinate). This is
        # an optimization component for finding paths for the personas' movement. 
        # self.address_tiles['<spawn_loc>bedroom-2-a'] == {(58, 9)}
        # self.address_tiles['double studio:recreation:pool table'] 
        #   == {(29, 14), (31, 11), (30, 14), (32, 11), ...}, 
        
        self.address_tiles = dict()
        for i in range(self.maze_height):
            for j in range(self.maze_width): 
                
                tile = self.tiles[i][j]

                if tile.collision:
                    continue

                address = tile.get_unique_name()

                if address in self.address_tiles: 
                    self.address_tiles[address].append(tile)
                else: 
                    self.address_tiles[address] = [tile]

        self.finder = SimplePathFinder(self.tiles)

        self.__visualize_grid_as_csv()

        f = self.address_tiles["the Ville:Moreno family's house:common room"][0]
        t = self.address_tiles["the Ville:artist's co-living space:Abigail Chen's room"][0]

        path = self.find_path(f, t)
        print(self.grid)

    def get_random_tile(self, tile) -> Tile:
        """
        returns a random tile from the maze and make sure it is not the same as the tile given
        """
        tiles = self.address_tiles[list(self.address_tiles)[random.randint(0, len(self.address_tiles) - 1)]]
        return tiles[random.randint(0, len(tiles) - 1)]
    
    def find_path(self, start: Tile, end: Tile) -> List[Tile]:
        """
        Calculates the path between two tiles.
        ARGS:
            start: start tile
            end: end tile
        RETURNS:
            List of tiles representing the path
        """

        path = self.finder.find_path(start, end)
        
        tiles = []
        for node in path:
            tiles += [self.get_tile(node.x, node.y)]

        return tiles
    
    @lru_cache(maxsize=1000)
    def _find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[GridNode]:
        start_node = self.grid.node(start[0], start[1])
        end_node = self.grid.node(end[0], end[1])

        return self.finder.find_path(start_node, end_node, self.grid)
        

    @lru_cache(maxsize=1000)
    def get_nearby_tiles(self, tile, vision_radius): 
        """
        Given a tile, we return all the tiles within a vision radius.

        X X X X X X X
        X X X X X X X
        X X X T X X X
        X X X X X X X
        X X X X X X X

        vision_radius = 1

        X X X X X X X
        X X N N N X X
        X X N T N X X
        X X N N N X X
        X X X X X X X

        returns Ns

        INPUT: 
            tile: A tile coordinate. 
            vision_radius: An integer representing the vision radius.
        OUTPUT:
            A set of tiles. 
        """

        # We need to get the tile coordinates of all the tiles within the vision 
        # radius. 
        nearby_tiles = list()

        for i in range(-vision_radius, vision_radius + 1):
            for j in range(-vision_radius, vision_radius + 1):
 
                if tile.x + i < 0 or tile.x + i >= self.maze_width or tile.y + j < 0 or tile.y + j >= self.maze_height:
                    continue
                
                nearby_tile = self.get_tile(tile.x + i, tile.y + j)

                if nearby_tile.is_walkable():
                    nearby_tiles += [nearby_tile]

        return nearby_tiles



    def __visualize_grid_as_csv(self, sep=";"):
        """
        Visualizes the grid as a csv file. 
        ARGS: 
        sep: separator for the csv file
        """
        out = ""
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                out += str(int(self.grid.node(i, j).walkable))+sep
            out += "\n"
        
        #write csv
        #with open("out.csv", "w") as f:
            #f.write(out)

    @staticmethod
    def convert_flat_list_to_2d_list(flat_list: List[str], width: int) -> List[List[str]]:
        """
        Converts a flat list to a 2d list. 
        ARGS:
        flat_list: list of strings
        width: width of the 2d list
        height: height of the 2d list
        RETURNS:
        2d list of strings
        """
        return [flat_list[i:i + width] for i in range(0, len(flat_list), width)]



    def load_csv(self):
        with open(self.csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.maze.append(row)

    @staticmethod
    def read_special_blocks(file_path: str) -> List[List[str]]:
        """
        Reads in a csv file to a list of list. If header is True, it returns a 
        tuple with (header row, all rows)
        ARGS:
        curr_file: path to the current csv file. 
        RETURNS: 
        List of list where the component lists are the rows of the file. 
        """
        with open(file_path) as file_handle:
            data_reader = csv.reader(file_handle, delimiter=",")
            return [[cell.strip() for cell in row] for row in data_reader]
        
    def print_maze(self):
        """ 
        prints the 2d list of tiles as a grid
        """
        for row in self.maze:
            print(row)

    def get_tile(self, x, y):
        return self.tiles[y][x]


if __name__ == "__main__":
    maze = Maze()
    print("maze created")