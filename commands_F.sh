conda activate GAI

# python main.py --setting=run_configs/default_config.py --n_gpu=1 --n_run=1
python main.py --setting=run_configs/default_config_2.py --n_gpu=1 --n_run=1
python main.py --setting=run_configs/default_config_3.py --n_gpu=1 --n_run=3
python main.py --setting=run_configs/default_config_trp.py --n_gpu=1 --n_run=1