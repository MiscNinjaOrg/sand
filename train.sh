accelerate launch --config_file "configs/simple_config.yaml" train.py \
--model_name "HuggingFaceH4/zephyr-7b-beta" \
--data_path "./alpaca_data.json" \
--cache_dir "/storage/ice1/4/7/nsemwal3/hf_cache" \
--output_dir "/storage/ice1/4/7/nsemwal3/model_out" \
--model_max_length 2048