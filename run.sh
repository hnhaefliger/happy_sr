set -e

init_kaggle=false
get_dataset=false
save_model=""

while [ -n "$1" ]; do # while loop starts
	case "$1" in

	"--init-kaggle") init_kaggle=true ;;
    "--get-dataset") get_dataset=true ;;
    "--save-model") $model_path="$2" ;;
    "--save-model") $model_path="$2" ;;

	*) echo "Option $1 not recognized" ;;

	esac

	shift

done

if [ $init_kaggle ]
then
    chmod +x ./happy_sr/scripts/colab_init_kaggle.sh
    ./happy_sr/scripts/colab_init_kaggle.sh
fi

if [ $get_dataset ]
then 
    chmod +x ./happy_sr/scripts/kaggle_cv_util.sh
    ./happy_sr/scripts/kaggle_cv_util.sh
    python ./happy_sr/scripts/format_labels.py
fi

python create_model.py