export PYTHONPATH=/workspace/AdaDistill:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train/train_AdaDistill.py
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
