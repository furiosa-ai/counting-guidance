## vanilla stable-diffusion
# mae/mse for counting
python run_config.py --cfg cfg/counting_sd.yaml --data data/prompts_v3.yaml 
python eval_count_acc.py --path "exp/counting_sd/" --data data/prompts_v3.yaml
python metrics/compute_clip_similarity.py --output_path "[exp/counting_sd/]"

# clip/blip for multiple objects
python run_config.py --cfg cfg/clip_sd.yaml --data data/prompts_multi_v3.yaml
python metrics/compute_clip_similarity.py --output_path "[exp/clip_sd/]"
python metrics/blip_captioning_and_clip_similarity.py --output_path "[exp/clip_sd/]"


## our counting guidance
# mae/mse for counting
python run_config.py --cfg cfg/counting_best.yaml --data data/prompts_v3.yaml 
python eval_count_acc.py --path "exp/counting_best/*" --data data/prompts_v3.yaml
python metrics/compute_clip_similarity.py --output_path "[exp/counting_best/*]"

# clip/blip for multiple objects
python run_config.py --cfg cfg/clip_best.yaml --data data/prompts_multi_v3.yaml
python metrics/compute_clip_similarity.py --output_path "[exp/clip_best/*]"
python metrics/blip_captioning_and_clip_similarity.py --output_path "[exp/clip_best/*]"


## our counting guidance (linear)
# mae/mse for counting
python run_config.py --cfg cfg/counting_linear_best.yaml --data data/prompts_v3.yaml 
python eval_count_acc.py --path "exp/counting_linear_best/*" --data data/prompts_v3.yaml
python metrics/compute_clip_similarity.py --output_path "[exp/counting_linear_best/*]"

# clip/blip for multiple objects (linear)
python run_config.py --cfg cfg/clip_linear_best.yaml --data data/prompts_multi_v3.yaml
python metrics/compute_clip_similarity.py --output_path "[exp/clip_linear_best/*]"
python metrics/blip_captioning_and_clip_similarity.py --output_path "[exp/clip_linear_best/*]"