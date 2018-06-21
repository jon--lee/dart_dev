CUDA_VISIBLE_DEVICES=$NUM python experiments/test_dart.py --envname Walker2d-v1 --t 500 --max_data 400 --num_evals 8 --update_period 300 --partition .10 --config net64-64
