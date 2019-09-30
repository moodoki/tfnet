#!/bin/bash

rates="2 4 8"
dataroot=${DATAROOT:-./data}
output_path=${OUTPUT_PATH:-./tfrecord_datasets}
txt_root=${TXTROOT:-./dataset_txt/vctk}
source_name=${NAME:-"VCTK-Corpus"}

for f in $txt_root/*.txt
do
    for rate in $rates
    do
        PYTHONPATH='./' python data_preparation/audioprep.py --fileslist="$f" \
            --dataroot="$dataroot" \
            --output_dir="$output_path" \
            --degrade_fn=downsample \
            --degrade_args=$rate \
            --source_name=$source_name
    done
done
