#!/bin/sh

set -e
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

init_kaggle=false
get_dataset=false
save_model=""
n_mels=64
hidden_dim=256
dropout=0.05
epochs=10
batch_size=16
learning_rate=1e-5
load=false
save=false

while [ -n "$1" ]; do # while loop starts
	case "$1" in

    "--init-kaggle") init_kaggle=true ;;
    "--get-dataset") get_dataset=true ;;
    "--save-model") 
        model_path="$2"
        shift ;;
    "--n-mels") 
        n_mels="$2"
        shift ;;
    "--hidden-dim")
        hidden_dim="$2"
        shift ;;
    "--dropout")
        dropout="$2"
        shift ;;
    "--epochs")
        epochs="$2"
        shift ;;
    "--batch-size")
        batch_size="$2"
        shift ;;
    "--lr")
        learning_rate="$2"
        shift ;;
    "--load")
        load="$2"
        shift ;;
    "--save")
        save="$2"
        shift ;;

	*) echo "Option $1 not recognized" ;;

	esac

	shift

done

if $init_kaggle
then
    chmod +x ./happy_sr/happy_sr/scripts/colab_init_kaggle.sh
    ./happy_sr/happy_sr/scripts/colab_init_kaggle.sh
fi

if $get_dataset
then 
    chmod +x ./happy_sr/happy_sr/scripts/kaggle_cv_util.sh
    ./happy_sr/happy_sr/scripts/kaggle_cv_util.sh
    python ./happy_sr/happy_sr/scripts/format_labels.py
fi

python3 happy_sr/create_model.py --n-mels $n_mels --hidden-dim $hidden_dim --dropout $dropout --epochs $epochs --batch-size $batch_size --lr $learning_rate --load $load --save $save
