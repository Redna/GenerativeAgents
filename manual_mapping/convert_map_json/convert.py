# json load the half_ville.tmj

# Path: manual_mapping/convert_map_json/convert.py
# Compare this snippet from manual_mapping/convert.py:
import csv
import json

def load_data(json_map):
    with open(json_map, 'r') as f:
        map = json.load(f)

    return map

def save_data(converted_maze, output_file):
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(converted_maze)


if __name__ == '__main__':

    layers = ['Arena Blocks', 'Sector Blocks', 'Object Interaction Blocks', 'Spawning Blocks', 'Collisions']
    file_names = ['arena_maze', 'sector_maze', 'game_object_maze', 'spawning_location_maze', 'collision_maze']
    suffix = '.csv'

    json_map = 'manual_mapping/convert_map_json/half_ville.tmj'
    map = load_data(json_map)

    for layer in map['layers']:
        if layer['name'] in layers:
            index = layers.index(layer['name'])
            file_name = file_names[index]
            output_file = 'manual_mapping/converted/' + file_name + suffix
            converted_maze = layer['data']
            save_data([converted_maze], output_file)

    print(map)