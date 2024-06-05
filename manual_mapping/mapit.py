


# load all csv files in the folder

import os
import csv
import json

path = 'manual_mapping/csv_files/'
files = os.listdir(path)


# open arena_maze.csv and base_arena_maze.csv

def map_numbers(from_maze, to_maze):
    from_to_mapping = {}
    to_from_mapping = {}

    for row_i, row in enumerate(from_maze):
        for col_i, col in enumerate(row):
            if row_i >= len(to_maze) or col_i >= len(to_maze[row_i]):
                continue

            to_number = to_maze[row_i][col_i]
            from_number = col

            from_to_mapping[from_number] = to_number
            to_from_mapping[to_number] = from_number
            

    return from_to_mapping, to_from_mapping

def convert_to_maze(from_to_mapping, from_maze):
    """ converts to maze, all not matching numbers are converted to 0"""
    maze = []
    for row in from_maze:
        new_row = []
        for number in row:
            if number in from_to_mapping:
                new_row.append(from_to_mapping[number])
            else:
                new_row.append('0')
        maze.append(new_row)
    
    return maze

def convert(from_path, to_path, output_path):
    with open(from_path, 'r') as f:
        reader = csv.reader(f)
        from_maze = list(reader)

    with open(to_path, 'r') as f:
        reader = csv.reader(f)
        to_maze = list(reader)

    number_mapping = map_numbers(from_maze, to_maze)
    converted_maze = convert_to_maze(number_mapping[1], from_maze)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(converted_maze)
    
    return number_mapping

if __name__ == '__main__':

    # convert arena

    mapping = convert(from_path='manual_mapping/csv_files/arena_maze.csv',
                to_path='manual_mapping/csv_files/base_arena_maze.csv',
                output_path='manual_mapping/csv_files/converted_base_arena_maze.csv')

    with open('manual_mapping/csv_files/arena_mapping.json', 'w') as f:
        json.dump(mapping[0], f)

    mapping = convert(from_path='manual_mapping/csv_files/sector_maze.csv',
            to_path='manual_mapping/csv_files/base_sector_maze.csv',
            output_path='manual_mapping/csv_files/converted_base_sector_maze.csv')
    
    with open('manual_mapping/csv_files/sector_mapping.json', 'w') as f:
        json.dump(mapping[0], f)
    
    mapping = convert(from_path='manual_mapping/csv_files/game_object_maze.csv',
            to_path='manual_mapping/csv_files/base_game_object_maze.csv',
            output_path='manual_mapping/csv_files/converted_base_game_object_maze.csv')

    with open('manual_mapping/csv_files/game_object_mapping.json', 'w') as f:
        json.dump(mapping[0], f)

    
    
    mapping = convert(from_path='manual_mapping/csv_files/spawning_location_maze.csv',
            to_path='manual_mapping/csv_files/base_spawning_location_maze.csv',
            output_path='manual_mapping/csv_files/converted_base_spawning_location_maze.csv')
    
    with open('manual_mapping/csv_files/spawning_location_mapping.json', 'w') as f:
        json.dump(mapping[0], f)
    
    mapping = convert(from_path='manual_mapping/csv_files/collision_maze.csv',
            to_path='manual_mapping/csv_files/base_collision_maze.csv',
            output_path='manual_mapping/csv_files/converted_base_collision_maze.csv')
    
    with open('manual_mapping/csv_files/collision_mapping.json', 'w') as f:
        json.dump(mapping[0], f)
    
    
    


