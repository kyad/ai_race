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
