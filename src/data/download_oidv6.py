import subprocess

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

classes_string = str()
for category in subset:
    classes_string += f"\"{category.replace('_', ' ')}\" "
print(classes_string)

# shell=True only for windows system
# subprocess.run(
#     f'oidv6 downloader en --dataset ../../data/raw/OpenImagesV6 --type_data train --classes "Swimming Pool" --limit 10 '
#     '--yes',
#     shell=True)
subprocess.run(
    f'oidv6 downloader en --dataset ../../data/raw/OpenImagesV6 --type_data all --classes {classes_string} --yes',
    shell=True)
