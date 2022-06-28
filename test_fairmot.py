import paddlehub as hub

import os

import cv2

# tracker = hub.Module(name="fairmot_dla34")

# # Read from a video file

# tracker.tracking('MOT16-14-raw.mp4', output_dir='mot_resulti_gpu', visualization=True, draw_threshold=0.5, use_gpu=True)

# #tracker.stream_mode(output_dir='', visualization=True, draw_threshold=0.5, use_gpu=True)

# print("++++" * 30)

# with tracker.stream_mode(output_dir='image_stream_output', visualization=True, draw_threshold=0.5, use_gpu=True):

#     img_list = os.listdir('mot_resulti_gpu/mot_outputs/MOT16-14-raw/')

#     inputs = []

#     for i, img in enumerate(img_list):

#         inputs.append(cv2.imread(os.path.join('mot_resulti_gpu/mot_outputs/MOT16-14-raw/', img)))

#     tracker.predict(inputs)
object_detector = hub.Module(name="yolov3_darknet53_coco2017")
frame_start= cv2.imread('MOT16-14-raw_raw/frame10.png')
results = object_detector.object_detection(images=[frame_start],use_gpu=True)
print(results)