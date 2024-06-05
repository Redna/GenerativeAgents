import csv
import json

def load_data(maze_file, mapping_file):
    with open(maze_file, 'r') as f:
        reader = csv.reader(f)
        arena_maze = list(reader)

    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    return arena_maze, mapping

def convert_data(arena_maze, mapping):
    converted_maze = []
    one_d_maze = []

    for row in arena_maze:
        converted_row = []
        for cell in row:
            new_value = 0
            if cell in mapping:
                new_value = mapping[cell]

            converted_row.append(new_value)
            one_d_maze.append(new_value)
        converted_maze.append(converted_row)

    return converted_maze, one_d_maze

def save_data(converted_maze, output_file):
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(converted_maze)

if __name__ == '__main__':
    maze_files = ['manual_mapping/to_convert/arena_maze.csv', 
                  'manual_mapping/to_convert/collision_maze.csv',
                  'manual_mapping/to_convert/sector_maze.csv',
                  'manual_mapping/to_convert/spawning_location_maze.csv',
                  'manual_mapping/to_convert/game_object_maze.csv']
    mapping_files = ['manual_mapping/mapping/arena_mapping.json',
                     'manual_mapping/mapping/collision_mapping.json',
                     'manual_mapping/mapping/sector_mapping.json',
                     'manual_mapping/mapping/spawning_location_mapping.json',
                     'manual_mapping/mapping/game_object_mapping.json']
    output_files = ['manual_mapping/converted/arena_maze.csv',
                    'manual_mapping/converted/collision_maze.csv',
                    'manual_mapping/converted/sector_maze.csv',
                    'manual_mapping/converted/spawning_location_maze.csv',
                    'manual_mapping/converted/game_object_maze.csv']

    for maze_file, mapping_file, output_file in zip(maze_files, mapping_files, output_files):
        arena_maze, mapping = load_data(maze_file, mapping_file)
        converted_maze, one_d_maze = convert_data(arena_maze, mapping)
        save_data(converted_maze, output_file)
        save_data([one_d_maze], output_file.replace('.csv', '_1d.csv'))