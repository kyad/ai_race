all:

checkpoints:
	python3 train.py \
		--test_interval 1 \
		--batch_size 128 \
		--data_csv $(HOME)/Images_from_rosbag/data-20201231.csv \
		--dataset_name data-20201231 \
		--model_name simplenet \
		--model simplenet \
		--model_ckpt_dir checkpoints \
		--model_ckpt_path_temp checkpoints/{}_{}_epoch{}.pth

simplenet_trt.pth: checkpoints
	python3 trt_conversion.py \
		--pretrained_model checkpoints/*.pth \
		--model simplenet \
		--trt_model $@

simplenet_trt_fp16.pth: checkpoints
	python3 trt_conversion.py \
		--fp16 \
		--pretrained_model checkpoints/*.pth \
		--model simplenet \
		--trt_model $@

20210105-simplenet-ckpt:
	python3 train.py \
		--test_interval 1 \
		--batch_size 20 \
		--data_csv $(HOME)/Images_from_rosbag/data-20210105/data-20210105.csv \
		--dataset_name data-20210105 \
		--model_name simplenet \
		--early_stopping 5 \
		--model simplenet \
		--model_ckpt_dir $@ \
		--model_ckpt_path_temp $@/{}_{}_epoch{}.pth

20210105-simplenet_trt.pth: 20210105-simplenet-ckpt
	python3 trt_conversion.py \
		--pretrained_model $</*.pth \
		--model simplenet \
		--trt_model $@

20210105-simplenet_trt_fp16.pth: 20210105-simplenet-ckpt
	python3 trt_conversion.py \
		--fp16 \
		--pretrained_model $</*.pth \
		--model simplenet \
		--trt_model $@

20210105a-simplenet-ckpt:
	python3 train.py \
		--test_interval 1 \
		--batch_size 20 \
		--data_csv $(HOME)/Images_from_rosbag/data-20210105/data-20210105_A.csv \
		--dataset_name data-20210105a \
		--early_stopping 5 \
		--model_name simplenet \
		--model simplenet \
		--model_ckpt_dir $@ \
		--model_ckpt_path_temp $@/{}_{}_epoch{}.pth

20210105a-simplenet_trt.pth: 20210105a-simplenet-ckpt
	python3 trt_conversion.py \
		--pretrained_model $</*.pth \
		--model simplenet \
		--trt_model $@

20210105a-simplenet_trt_fp16.pth: 20210105a-simplenet-ckpt
	python3 trt_conversion.py \
		--fp16 \
		--pretrained_model $</*.pth \
		--model simplenet \
		--trt_model $@

20210105a-simplenet2-ckpt:
	python3 train.py \
		--test_interval 1 \
		--batch_size 20 \
		--data_csv $(HOME)/Images_from_rosbag/data-20210105/data-20210105_A.csv \
		--dataset_name data-20210105a \
		--early_stopping 5 \
		--model_name simplenet2 \
		--model simplenet2 \
		--model_ckpt_dir $@ \
		--model_ckpt_path_temp $@/{}_{}_trial{}_epoch{}.pth

20210105a-simplenet2_trt_fp16.pth: 20210105a-simplenet2-ckpt
	python3 trt_conversion.py \
		--fp16 \
		--pretrained_model $</*.pth \
		--model simplenet2 \
		--trt_model $@

vit-ckpt:
	python3 train.py \
		--test_interval 1 \
		--batch_size 128 \
		--data_csv $(HOME)/Images_from_rosbag/data-20201231.csv \
		--dataset_name data-20201231 \
		--model_name vit \
		--model vit \
		--model_ckpt_dir $@ \
		--model_ckpt_path_temp $@/{}_{}_epoch{}.pth

vit_trt.pth: vit-ckpt
	python3 trt_conversion.py \
		--pretrained_model $</*.pth \
		--model vit \
		--trt_model $@

vit2-ckpt:
	python3 train.py \
		--test_interval 1 \
		--batch_size 128 \
		--data_csv $(HOME)/Images_from_rosbag/data-20201231.csv \
		--dataset_name data-20201231 \
		--model_name vit2 \
		--model vit2 \
		--model_ckpt_dir $@ \
		--model_ckpt_path_temp $@/{}_{}_epoch{}.pth

vit2_trt.pth: vit2-ckpt
	python3 trt_conversion.py \
		--pretrained_model $</*.pth \
		--model vit2 \
		--trt_model $@

20210301-vit2-ckpt:
	python3 train.py \
		--test_interval 1 \
		--batch_size 128 \
		--data_csv $(HOME)/Images_from_rosbag/data-20210301/data-20210301.csv \
		--dataset_name data-20210301 \
		--model_name vit2 \
		--model vit2 \
		--model_ckpt_dir $@ \
		--model_ckpt_path_temp $@/{}_{}_trial{}_epoch{}.pth
