#!/opt/homebrew/bin/bash
counter=0
num_runs=3 # number runs for each combination of hyperparams
algs=(spectral)
total_count=$((${#algs[@]} * $num_runs))

declare -A args=(
    ["backbone"]="relu"
    ["optim"]="sgd"
    ["aug"]="normal"
    ["num_feats"]="5"
    ["feat_dim"]="5"
    ["bs"]="256"
    ["lr"]="3e-4"
    ["sched"]="step"
    ["epochs"]="100"
    ["emb_dim"]="5"
    # ["loss_type"]="mce"
    ["save_dir"]="./outputs/normal/relu/spectral"
)

# Appends arguments to command
append_args() {
    local cmd="$1"
    for key in "${!args[@]}"; do
        cmd+=" --$key ${args[$key]}"
    done
    echo "$cmd"
}

if [ "$1" == "--debug" ]; then
    echo "Running in debug mode... (only using alg ${algs[0]})"
    cmd="python3 -m debugpy --listen 5678 --wait-for-client pretrain.py --alg ${algs[0]}"
    cmd=$(append_args "$cmd")
    echo "Debug command: $cmd"
    echo "Executing command..."
    eval $cmd
    echo "Command execution completed with exit code: $?"
else
    echo "Running normally..."
    for i in $(seq 1 $num_runs)
    do
        for alg in ${algs[@]}
        do
            ((counter++))
            echo "Run $counter of $total_count"
            cmd="python3 pretrain.py --alg $alg"
            cmd=$(append_args "$cmd")
            echo "Running command $cmd"
            eval $cmd
            # python3 pretrain.py --alg $alg --backbone $backbone --optim $optim --aug $aug --num_feats $num_feats --feat_dim $feat_dim --bs $bs --lr $lr --sched $sched --epochs $epochs --emb_dim $emb_dim --loss_type $loss_type --save_dir $save_dir
        done
    done
fi


# './outputs/normal_hparam_tuning2/normal/spectral_linear_sgd_emb_dim=5_aug=normal_lr=0.0003_sched=step_run1',


# ./outputs/normal_hparam_tuning2


# defaults in pretrain.py:
# n (2 ** 16)
# num_feats 5
# feat_dim 5
# embd 10
# epochs 100
# bs 256
# lr 1e-3
# wd 1e-5
# momentum 0.9