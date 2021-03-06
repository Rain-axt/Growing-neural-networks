data_path=''  # training data path
eval_path=''  # selection set path
tiny_dev=''  # dev set path
test_path=''  # test set path
data_dump_path=''  # segmented dumped selection data
tune_eval_path=''  # eval data in branch selection
local_path=''
kernel_dim=9
max_layers=4
tree_volume=75
batch_size=32
lr=2e-3
epoch=1
steps_per_epoch=40000
load_path=''  # load model from this directory
dim_shift=6
input_dim='32,32'  # for cifar10: 32,32, for mnist: 28,28
dim_span=3

model='none'  # if train model from scratch, set to none
param='none'
step='test'
end_step='end'
make_initial=1  # for tuning mode, set to 1, for election mode, set to 0

python3 -u train_dynamic_network.py --local_path=$local_path \
	--data_path=$data_path \
	--eval_path=$eval_path \
	--test_path=$test_path \
	--data_dump_path=$data_dump_path \
	--tune_eval_path=$tune_eval_path \
	--kernel_dim=$kernel_dim \
	--max_layers=$max_layers \
	--tree_volume=$tree_volume \
	--batch_size=$batch_size \
	--lr=$lr \
	--epoch=$epoch \
	--steps_per_epoch=$steps_per_epoch \
	--tiny_dev=$tiny_dev \
	--load_path=$load_path \
	--dim_shift=$dim_shift \
	--input_dim=$input_dim \
	--dim_span=$dim_span \
	--model=$model \
	--param=$param \
	--step=$step \
	--end_step=$end_step \
	--make_initial=$make_initial
