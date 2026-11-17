nohup bash tools/dist_train.sh configs/rtm/rtm_voc_split.py  3 >130150.out 2>&1 &
wait
nohup bash tools/dist_train.sh configs/rtm/rtm_voc_base.py  3 >130150.out 2>&1 &