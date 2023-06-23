from shutil import copy
import os



coco_names= [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
# traffic_names=[ 'traffic light', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'dog', 'cat', 'stop sign', 'fire hydrant', 'train' ]
traffic_names=['person', 'bicycle', 'car', 'motorcycle', 'bus','train', 'truck', 'traffic light','fire hydrant','stop sign','cat', 'dog']
print(len(traffic_names))
index_list=[]
for index, names in enumerate(coco_names):
    if names in traffic_names:
        print(index, names)
        index_list.append(index)

# list_index=[0,1,2,3,5,6,7,9,10,15,16]
list_index=["0","1","2","3","5","6","7","9","10","15","16"]

label_base_path = r'C:\Users\User\Documents\machine_learning\yolov7\data\coco\labels\train2017'
label_destination_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco_traffic\train\labels'

image_base_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco\images\train2017'
image_destination_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco_traffic\train\images'

def copy_labels(source,destination):
    try:
        copy(source,destination)
    except: pass
    
def copy_images(source,destination):
    try:
        copy(source,destination)
    except FileNotFoundError:
        return source + " not found"
        print("File not found")


filename='000000425481.txt'
with open(os.path.join(label_base_path,filename)) as f:
    file_number = filename.split('.')[0]
    print(file_number)
    lines = f.readlines()
    for line_index,line in enumerate(lines):
        print("\n")
        single_lines=line.split(' ')
        print(single_lines[0])
        print(True if single_lines[0] in list_index else False)

file_in_list_index=[]
for file in os.listdir(label_base_path):
    print(file)
    with open(os.path.join(label_base_path,file)) as f:
        file_number = file.split('.')[0]
        lines = f.readlines()
        for line_index,line in enumerate(lines):
            single_lines=line.split(' ')
            # print(single_lines[0])
            # print(True if single_lines[0] in list_index else False)
            file_in_list_index.append(True if single_lines[0] in list_index else False)
    if 'True' in file_in_list_index:
        if os.path.exists(os.path.join(image_base_path,file,'.jpg')):
            copy_labels(os.path.join(label_base_path,file), os.path.join(label_destination_path,file))
            copy_images(os.path.join(image_base_path,file_number,'.jpg'), os.path.join(image_destination_path,file_number,'.jpg'))
    file_in_list_index.clear()
                    