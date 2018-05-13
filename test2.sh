CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly6
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly7
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly8

CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly6
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly7
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly8
