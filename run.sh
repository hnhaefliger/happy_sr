#!/bin/sh

set -e

init_kaggle=false
get_dataset=false
save_model=""
n_mels = 64
hidden_dim = 256
dropout = 0.05
epochs = 10

while [ -n "$1" ]; do # while loop starts
	case "$1" in

	"--init-kaggle") init_kaggle=true ;;
    "--get-dataset") get_dataset=true ;;
    "--save-model") $model_path="$2" ;;
    "--n-mels") $n_mels="$2" ;;
    "--hidden-dim") $hidden_dim="$2" ;;
    "--dropout") $dropout="$2" ;;
    "--epochs") $epochs="$2" ;;

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

python happy_sr/create_model.py --n-mels $n_mels --hidden-dim $hidden_dim --dropout $dropout --epochs $epochs