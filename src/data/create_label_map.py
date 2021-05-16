import argparse
import os

subset = ["Toilet",
          "Swimming_pool",
          "Bed",
          "Billiard_table",
          "Sink",
          "Fountain",
          "Oven",
          "Ceiling_fan",
          "Television",
          "Microwave_oven",
          "Gas_stove",
          "Refrigerator",
          "Kitchen_&_dining_room_table",
          "Washing_machine",
          "Bathtub",
          "Stairs",
          "Fireplace",
          "Pillow",
          "Mirror",
          "Shower",
          "Couch",
          "Countertop",
          "Coffeemaker",
          "Dishwasher",
          "Sofa_bed",
          "Tree_house",
          "Towel",
          "Porch",
          "Wine_rack",
          "Jacuzzi"]

def save_label_map(label_map_path, data):
    with open(label_map_path, 'w+') as f:
        for i in range(len(data)):
            line = "item {\nid: " + str(i + 1) + "\nname: '" + data[i] + "'\n}\n"
            f.write(line)

if __name__ == '__main__':
    trainable_classes_file = [cat.replace("_", " ") for cat in subset]
    label_map_path = "label_map.pbtxt"
    save_label_map(label_map_path, trainable_classes_file)
