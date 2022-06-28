cd jde
hub install jde_darknet53
cd ..
hub run jde_darknet53 --video_stream MOT16-14-raw.mp4 --use_gpu  --visualization
