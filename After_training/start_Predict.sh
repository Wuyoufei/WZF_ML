suffix=$1
rrun=$2
cwd=`pwd`
echo 'Please ensure the requirements are what you need: ';echo
sed -n '6,9p' run_Predict.sh;echo

if [ -z $suffix ]; then
    read -p 'please input net/suffix for this prediction: ' suffix 
    echo "suffix is '$suffix'"
else
    echo "suffix is '$suffix'"
fi

while [ -e "./output/figs/conf_${suffix}.pdf" ]
do
    read -p "suffix '${suffix}' has been used, override it?(y/n): " option
    case ${option} in
        "y")
            echo "override suffix '${suffix}' confirmed"
            break
            ;;
        "n")
            read -p "reset your suffix: " suffix
            ;;
        *)
            echo 'please type y or n'
            ;;
    esac
done
        



sed -i "10c #SBATCH --output=${cwd}/output/my_log/predict_${suffix}.log" run_Predict.sh
sed -i "45c python -u script_Predict/apply_and_predict.py ${suffix}" run_Predict.sh

if [ "$rrun" == l ]; then
    echo 'running code locally(for test usually)'
    python -u script_Predict/apply_and_predict.py ${suffix}
    exit 0
elif [ "$rrun" == s ]; then
    sbatch run_Predict.sh
    exit 0
fi



while [ "$rrun" != l -a "$rrun" != s ]
do
    read -p 'run local or slurm (l/s): ' rrun
    if [ "$rrun" == l ]; then
        echo 'running code locally(for prediction with low statistic)'
        python -u script_Predict/apply_and_predict.py ${suffix}
        exit 0
    elif [ "$rrun" == s ]; then
        sbatch run_Predict.sh
        exit 0
    fi
done












