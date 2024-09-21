import yaml
# import ultralytics
# ultralytics.checks()

from ultralytics import YOLO

# Load the checkpoint from the 40th epoch
model = YOLO("yolov8n.pt")
names = model.model.names

# Step 2: Read the .yaml file
yaml_file_path = '/home/javad/code/vehicle_speed_detection2/python/server/fine-tune/yolo-dataset/dataset.yaml'

# Step 3: Modify the specific line
yaml_content = {}
yaml_content['path'] = '/home/javad/code/vehicle_speed_detection2/python/server/fine-tune/yolo-dataset/'
yaml_content['train'] = 'train'
yaml_content['val'] = 'val'
yaml_content['test'] = 'test'

yaml_content['names'] = names

# Step 4: Write back the changes
with open(yaml_file_path, 'w') as file:
    yaml.dump(yaml_content, file)


# Resume training for the remaining epochs
results = model.train(data=yaml_file_path, epochs=4)
