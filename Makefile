
DOWNSAMPLE_RATES=2 4 8
OUT_DIR=./logdir
DATA_DIR=./tfrecord_datasets

k2017_vctk_p225=$(addprefix $(OUT_DIR)/kuleshov2017/vctk-p225/ds,$(DOWNSAMPLE_RATES))
k2017_vctk_m=$(addprefix $(OUT_DIR)/kuleshov2017/vctk-multispeaker/ds,$(DOWNSAMPLE_RATES))

tfnet2018_vctk_p225=$(addprefix $(OUT_DIR)/tfnet2018/vctk-p225/ds,$(DOWNSAMPLE_RATES))
tfnet2018_vctk_m=$(addprefix $(OUT_DIR)/tfnet2018/vctk-multispeaker/ds,$(DOWNSAMPLE_RATES))

all:

all_baselines: $(OUT_DIR)/kuleshov2017 $(OUT_DIR)/tfnet

$(OUT_DIR)/kuleshov2017: $(k2017_vctk_p225) $(k2017_vctk_m)

k2017_p225: $(k2017_vctk_p225)

k2017_multispeaker: $(k2017_vctk_m)

$(k2017_vctk_p225): presets-flags/kuleshov-et-al-2017.flags
	python train.py --flagfile=$< \
		--trainset=$(DATA_DIR)/vctk-p225-train-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--testset=$(DATA_DIR)/vctk-p225-val-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--downsample_rate=`echo -n $@|tail -c -1`\
		--model_dir=$@

$(k2017_vctk_m): presets-flags/kuleshov-et-al-2017.flags
	python train.py --flagfile=$< \
		--trainset=$(DATA_DIR)/vctk-multispeaker-train-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--testset=$(DATA_DIR)/vctk-multispeaker-val-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--downsample_rate=`echo -n $@|tail -c -1`\
		--model_dir=$@

tfnet_baselines: $(OUT_DIR)/tfnet2018

$(OUT_DIR)/tfnet2018: $(OUT_DIR)/tfnet2018/vctk-p225 $(OUT_DIR)/tfnet2018/vctk-multispeaker

$(OUT_DIR)/tfnet2018/vctk-p225: tfnet_baseline_p225

$(OUT_DIR)/tfnet2018/vctk-multispeaker: tfnet_baseline_multispeaker

tfnet_baseline_p225: $(tfnet2018_vctk_p225)

tfnet_baseline_multispeaker: $(tfnet2018_vctk_m)

$(tfnet2018_vctk_p225): presets-flags/tfnet-2018.flags
	python train.py --flagfile=$< \
		--trainset=$(DATA_DIR)/vctk-p225-train-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--testset=$(DATA_DIR)/vctk-p225-val-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--downsample_rate=`echo -n $@|tail -c -1`\
		--model_dir=$@

$(tfnet2018_vctk_m): presets-flags/tfnet-2018.flags
	python train.py --flagfile=$< \
		--trainset=$(DATA_DIR)/vctk-multispeaker-train-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--testset=$(DATA_DIR)/vctk-multispeaker-val-16000-downsample-`echo -n $@|tail -c -1`.tfrecord\
		--downsample_rate=`echo -n $@|tail -c -1`\
		--model_dir=$@


clean_baselines:
	rm -rf $(OUT_DIR)/kuleshov2017

.PHONY: all all_baselines clean_baselines tfnet_baselines tfnet_baseline_p225 tfnet_baseline_multispeaker
