cd ..

subtask="boolean_expressions"

python cli.py \
 --do_train --do_validate --do_test \
 --trainer pe2 --backtrack --resume \
 --meta_prompts_dir meta_prompts/pe2 \
 --model zeroshotcot \
 --task bbh --subtask ${subtask} \
 --data_dir data/bbh/${subtask} \
 --output_dir output/bbh/${subtask}/pe2/ \
 --task_model openai_gpt35_turbo_instruct \
 --optim_model openai_gpt4_turbo \
 --n_beam 4 \
 --n_expand 4 \
 --train_steps 4 \
 --init_temperature 0.7 \
 --init_method file \
 --init_prompt_file prompt.md;
