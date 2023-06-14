# ftcn
srun python -m torch.distributed.launch --nproc_per_node=4 --master_port=45000 --use_env main_ftcn.py --batch_size 8 --output_dir ./checkpoints/ftcn/
# ftcn
srun python -m torch.distributed.launch --nproc_per_node=4 --master_port=45000 --use_env main_ftcn.py --batch_size 8 --output_dir ./checkpoints/ftcn/
