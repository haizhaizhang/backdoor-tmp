nohup bash tools/dist_train.sh configs/yolov5/ghost/yolov5_s_0.5.py 3 >0.5.out 2>&1 &
wait
nohup bash tools/dist_train.sh configs/yolov5/ghost/yolov5_s_0.3.py 3 >0.3.out 2>&1 &




