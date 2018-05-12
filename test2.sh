CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly2
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly3
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly5
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly10
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config poly20

CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly2
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly3
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly5
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly10
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config poly20
