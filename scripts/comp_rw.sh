model_arr=('meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-hf' 'mistralai/Mistral-7B-Instruct-v0.1')
task_arr=('sst2' 'boolq' 'qqp' 'wic' 'rte' 'mnli' 'agnews' 'arc')
n=4

seed_arr=(0 1 2 3 4)
model=${model_arr[0]}
bs=8


for form in {1..3}
do
    for task in "${task_arr[@]}"
    do
        python decompose.py --format ${form} --batch_size ${bs} --model_name ${model} --dataset ${task} --n_shots $n --seed_list ${seed_arr[@]}
    done
done

python train_components.py
