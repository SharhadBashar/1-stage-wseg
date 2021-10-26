#!/bin/bash

## Your values here:
#
DS=pascal_voc
EXP=baseline_101
RUN_ID=v01_size_1e-1
#
##

#
# Script
#
now=$(date)
LOG_DIR=logs/${DS}/${EXP}
CMD="python train.py --dataset $DS --cfg configs/voc_resnet101.yaml --exp $EXP --run $RUN_ID --set NET.MODEL bsl TRAIN.NUM_EPOCHS 6"
LOG_FILE=$LOG_DIR/${RUN_ID}.log

if [ ! -d "$LOG_DIR" ]; then
  echo "Creating directory $LOG_DIR $now"
  mkdir -p $LOG_DIR
fi

echo $CMD
echo "LOG: $LOG_FILE"

nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
