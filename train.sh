mkdir -p results
mkdir -p saves



model_names=('tsai_rnnplus' 'tsai_TCN')
batch_sizes=(32 64)                
learning_rates=(1e-1 1e-2)
epochss=(100)                      
label_smoothings=(0.0 0.1)     
lag_sts=(0)
lag_ed=0
min_n=0
max_ns=(0)
wds=(1e-2 1e-3)
mode='all'
dataset_names=('Cancer' 'Dyslipidemia') # '' Dyslipidemia
augmentations=('no')
mixups=('yes' 'no')




GPUS=(0 1 2 3 4 5 6 7 8 9)


total_iters=$((${#epochss[@]} * ${#mixups[@]} * ${#dataset_names[@]}* ${#wds[@]} * ${#lag_sts[@]} * ${#batch_sizes[@]} * ${#learning_rates[@]} * ${#model_names[@]} * ${#label_smoothings[@]} * ${#augmentations[@]} * ${#max_ns[@]}))
current_iter=0
start_time=$SECONDS



gpu_idx=0

for epochs in "${epochss[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for lag_st in "${lag_sts[@]}"; do
            for bsz in "${batch_sizes[@]}"; do
                for lr in "${learning_rates[@]}"; do
                    for model_name in "${model_names[@]}"; do
                        for label_smoothing in "${label_smoothings[@]}"; do
                            for augmentation in "${augmentations[@]}"; do
                                for max_n in "${max_ns[@]}"; do
                                    for wd in "${wds[@]}"; do
                                        for mixup in "${mixups[@]}"; do

                                            current_time=$(date +%T)
                                            elapsed_time=$((SECONDS - start_time))
                                            hours=$((elapsed_time / 3600))
                                            minutes=$(( (elapsed_time % 3600) / 60))
                                            seconds=$((elapsed_time % 60))
                                            current_iter=$((current_iter + 1))

                                            echo "Current Time: $current_time"
                                            echo "Elapsed Time: $hours hour(s) $minutes minute(s) $seconds second(s)"
                                            echo "Progress: $current_iter / $total_iters"

                                            CUDA_VISIBLE_DEVICES="${GPUS[gpu_idx]}" \
                                            python train.py \
                                            --bsz "$bsz" \
                                            --lag-st "$lag_st" \
                                            --lag-ed "$lag_ed" \
                                            --epochs "$epochs" \
                                            --lr "$lr" \
                                            --wd "$wd" \
                                            --mode "$mode" \
                                            --model_name "$model_name" \
                                            --label_smoothing "$label_smoothing" \
                                            --augmentation "$augmentation" \
                                            --min_n "$min_n" \
                                            --max_n "$max_n" \
                                            --wd "$wd" \
                                            --mixup "$mixup" \
                                            --dataset_name "$dataset_name" &

                                            gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))

                                            echo "$gpu_idx"

                                            if [[ $gpu_idx -eq 0 ]]; then
                                                wait
                                            fi
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
wait  
