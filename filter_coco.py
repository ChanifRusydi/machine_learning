from shutil import copy
import os

coco_names= [ 'person', 'bicycle','car', 'motorcycle', 'airplane', 'bus', 'train', 'truck','boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
         'horse', 'sheep', 'cow','elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
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

#convert index to string
list_index=[]
for index in index_list:
    list_index.append(str(index))
    
# list_index=[0,1,2,3,5,6,7,9,10,11,15,16]
list_index=["0","1","2","3","5","6","7","9","10","11","15","16"]

label_base_path = r'C:\Users\User\Documents\machine_learning\yolov7\data\coco\labels\train2017'
label_destination_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco_traffic\train\labels'

image_base_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco\images\train2017'
image_destination_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco_traffic\train\images'

# label_base_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco\labels\val2017'
# label_destination_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco_traffic\val\labels'

# image_base_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco\images\val2017'
# image_destination_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\coco_traffic\val\images'

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
# filename='000000425481.txt'
# file_number = filename.split('.')[0]
# jpg_file=file_number+'.jpg'
# print(os.path.join(image_base_path,jpg_file))
# copy_labels(os.path.join(label_base_path,filename), os.path.join(label_destination_path,filename))
# copy_images(os.path.join(image_base_path,jpg_file), os.path.join(image_destination_path,jpg_file))
# print(os.listdir(label_base_path))
# filename='000000425481.txt'
# with open(os.path.join(label_base_path,filename)) as f:
#     file_number = filename.split('.')[0]
#     print(file_number)
#     lines = f.readlines()
#     for line_index,line in enumerate(lines):
#         print("\n")
#         single_lines=line.split(' ')
#         print(single_lines[0])
#         print(True if single_lines[0] in list_index else False)

# file_in_list_index=[]
# for file_index,file in enumerate(os.listdir(label_base_path)):
#     print('\n',file)
#     with open(os.path.join(label_base_path,file)) as f:
#         file_number = file.split('.')[0]
#         lines = f.readlines()
#         for line_index,line in enumerate(lines):
#             single_lines=line.split(' ')
#             # print(single_lines[0])
#             print(single_lines[0],True if single_lines[0] in list_index else False)
#             file_in_list_index.append(True if single_lines[0] in list_index else False)
#     jpg_file=file_number+'.jpg'   
#     if True in file_in_list_index:
#         if os.path.exists(os.path.join(image_base_path,jpg_file)):
#             copy_labels(os.path.join(label_base_path,file), os.path.join(label_destination_path,file))
#             jpg_file=file_number+'.jpg'
#             copy_images(os.path.join(image_base_path,jpg_file), os.path.join(image_destination_path,jpg_file))
#     file_in_list_index.clear()
#     if file_index==10:
#         break

# test_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\test'
# for file_index,file in enumerate(os.listdir(label_destination_path)):
#     print('\n',file)
#     with open(os.path.join(label_destination_path,file),'r') as f:
#         file_number = file.split('.')[0]
#         lines = f.readlines()
#         print(lines)
#     with open(os.path.join(label_destination_path,file),'w') as file_write:
#         for line_index,line in enumerate(lines):
#             print(line)
#             single_lines=line.split(' ')
#             print(single_lines[0])
#             if single_lines[0] in list_index:
#                 file_write.write(line)



# test_path=r'C:\Users\User\Documents\machine_learning\yolov7\data\test'
# for file_index,file in enumerate(os.listdir(label_destination_path)):
#     print('\n',file)
#     with open(os.path.join(label_destination_path,file),'r') as f:
#         file_number = file.split('.')[0]
#         lines = f.readlines()
#         print(lines)
#     with open(os.path.join(label_destination_path,file),'w') as file_write:
#         for line_index,line in enumerate(lines):
#             print(line)
#             single_lines=line.split(' ')
#             print(single_lines[0])
#             if single_lines[0]=="11":
#                 single_lines[0]="9"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             elif single_lines[0]=="5":
#                 single_lines[0]="4"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             elif single_lines[0]=="6":
#                 single_lines[0]="5"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             elif single_lines[0]=="7":
#                 single_lines[0]="6"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             elif single_lines[0]=="9":
#                 single_lines[0]="7"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             elif single_lines[0]=="10":
#                 single_lines[0]="8"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             elif single_lines[0]=="15":
#                 single_lines[0]="10"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             elif single_lines[0]=="16":
#                 single_lines[0]="11"
#                 new_line=" ".join(single_lines)
#                 print(new_line)
#                 file_write.write(new_line)
#             else:
#                 file_write.write(line)        