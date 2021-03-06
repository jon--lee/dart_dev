CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --update_period 50 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --update_period 300 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dart.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --update_period 300 --partition .10 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_iso.py --envname Hopper-v1 --t 500 --max_data 400 --num_evals 8 --scale 1.0 --config linear


CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --update_period 50 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --update_period 300 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dart.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --update_period 300 --partition .10 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_iso.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --scale 1.0 --config linear


CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname HalfCheetah-v1 --t 500 --max_data 400 --num_evals 8 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname HalfCheetah-v1 --t 500 --max_data 400 --num_evals 8 --update_period 50 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname HalfCheetah-v1 --t 500 --max_data 400 --num_evals 8 --update_period 300 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dart.py --envname HalfCheetah-v1 --t 500 --max_data 400 --num_evals 8 --update_period 300 --partition .10 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_iso.py --envname HalfCheetah-v1 --t 500 --max_data 400 --num_evals 8 --scale 1.0 --config linear


CUDA_VISIBLE_DEVICES=$NUM python experiments/test_bc.py --envname Humanoid-v1 --t 500 --max_data 10000 --num_evals 8 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname Humanoid-v1 --t 500 --max_data 10000 --num_evals 8 --update_period 1000 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dagger.py --envname Humanoid-v1 --t 500 --max_data 10000 --num_evals 8 --update_period 200 --beta .5 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dart.py --envname Humanoid-v1 --t 500 --max_data 10000 --num_evals 8 --update_period 1000 --partition .10 --config linear
CUDA_VISIBLE_DEVICES=$NUM python experiments/test_iso.py --envname Humanoid-v1 --t 500 --max_data 10000 --num_evals 8 --scale 1.0 --config linear

