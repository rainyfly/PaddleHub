import os
import re
import sys
import shutil

# logfile = open('mismatch_versionmodule.txt', 'a')

# object_detection_modules = ['faster_rcnn_resnet50_coco2017', 'faster_rcnn_resnet50_fpn_coco2017',
#  'yolov3_darknet53_pedestrian', 'ssd_mobilenet_v1_pascal', 'yolov3_resnet34_coco2017',
#  'ssd_vgg16_512_coco2017', 'yolov3_resnet50_vd_coco2017', 'yolov3_darknet53_vehicles', 'yolov3_darknet53_coco2017',
#  'faster_rcnn_resnet50_fpn_venus', 'yolov3_mobilenet_v1_coco2017', 'yolov3_darknet53_venus']
# object_detection_modules = ['faster_rcnn_resnet50_fpn_coco2017']

face_dection_modules = ['pyramidbox_lite_server', 'pyramidbox_lite_server_mask',
'ultra_light_fast_generic_face_detector_1mb_640', 'pyramidbox_lite_mobile',
'pyramidbox_lite_mobile_mask', 'ultra_light_fast_generic_face_detector_1mb_320', 'pyramidbox_face_detection']

# depth_estimation_modules = ['MiDaS_Large']


for modulename in object_detection_modules:
    pattern1 = r'hub install {}==(\d.\d.\d)'.format(modulename)
    pattern2 = r'version="(\d.\d.\d)"'
    with open(os.path.join('modules/image/object_detection', modulename, 'module.py')) as f:
        for res in f.readlines():
            match = re.search(pattern2, res)
            if match:
                code_version = match[1]
                break
    
    with open(os.path.join('modules/image/object_detection', modulename, 'README.md')) as f:
        for res in f.readlines():
            match = re.search(pattern1, res)
            if match:
                readme_version = match[1]
                break
    if code_version != readme_version:
        print(modulename)
        logfile.write(modulename)
    else:
        continue
    print('开始处理模块{}'.format(modulename))
    with open(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'module.py')) as f:
        with open(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'newmodule.py'), 'w') as nf:
            for res in f.readlines():
                match = re.search(pattern2, res)
                if match:
                    res = res.replace(code_version, readme_version)
                nf.write(res)
    shutil.move(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'newmodule.py'), os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'module.py')) #将新module.py覆盖原来的
    shutil.copy(os.path.join('modules/image/object_detection', modulename, 'README.md'), os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename)) #先复制README更新.paddlehub
    for filename in os.listdir(os.path.join('modules/image/object_detection', modulename)):
        shutil.copy(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename,filename), os.path.join('modules/image/object_detection', modulename)) # 将文件复制回repo
    
    os.system('cd {} && tar czvf {} {}'.format('/ssd5/chenjian26/.paddlehub/modules/','{}_{}.tar.gz'.format(modulename, readme_version), modulename))
    os.system('cd {} && mv {} {}'.format('/ssd5/chenjian26/.paddlehub/modules/', '{}_{}.tar.gz'.format(modulename, readme_version),'/ssd5/chenjian26/HubPackages/'))

    
                


# for modulename in face_dection_modules:
#     pattern1 = r'hub install {}==(\d.\d.\d)'.format(modulename)
#     pattern2 = r'version="(\d.\d.\d)"'
#     with open(os.path.join('modules/image/face_detection', modulename, 'module.py')) as f:
#         for res in f.readlines():
#             match = re.search(pattern2, res)
#             if match:
#                 code_version = match[1]
#                 break
    
#     with open(os.path.join('modules/image/face_detection', modulename, 'README.md')) as f:
#         for res in f.readlines():
#             match = re.search(pattern1, res)
#             if match:
#                 readme_version = match[1]
#                 break
#     if code_version != readme_version:
#         print(modulename)
#         logfile.write(modulename + '\n')
#     else:
#         continue
#     print('开始处理模块{}'.format(modulename))
#     with open(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'module.py')) as f:
#         with open(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'newmodule.py'), 'w') as nf:
#             for res in f.readlines():
#                 match = re.search(pattern2, res)
#                 if match:
#                     res = res.replace(code_version, readme_version)
#                 nf.write(res)
#     shutil.move(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'newmodule.py'), os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename, 'module.py')) #将新module.py覆盖原来的
#     shutil.copy(os.path.join('modules/image/face_detection', modulename, 'README.md'), os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename)) #先复制README更新.paddlehub
#     for filename in os.listdir(os.path.join('modules/image/face_detection', modulename)):
#         shutil.copy(os.path.join('/ssd5/chenjian26/.paddlehub/modules/', modulename,filename), os.path.join('modules/image/face_detection', modulename)) # 将文件复制回repo
    
    # os.system('cd {} && tar czvf {} {}'.format('/ssd5/chenjian26/.paddlehub/modules/','{}_{}.tar.gz'.format(modulename, readme_version), modulename))
    # os.system('cd {} && mv {} {}'.format('/ssd5/chenjian26/.paddlehub/modules/', '{}_{}.tar.gz'.format(modulename, readme_version),'/ssd5/chenjian26/HubPackages'))




