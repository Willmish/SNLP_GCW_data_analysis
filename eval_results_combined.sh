#! /bin/bash

BASE_PATH=`pwd`

# Paths for Harmful benchmarking results
EVAL_HARMFULNESS_PATH=$BASE_PATH/eval_harmfulness
BATCH_EVAL_PATH_HARMFULNESS=$EVAL_HARMFULNESS_PATH/batch_unlearning
SEQ_EVAL_PATH_HARMFULNESS=$EVAL_HARMFULNESS_PATH/sequential_unlearning
LLM_UNLEARN_EVAL_PATH_HARMFULNESS=$EVAL_HARMFULNESS_PATH/llm_unlearning_reproduced
BATCH_SCALED_LR_EVAL_PATH_HARMFULNESS=$EVAL_HARMFULNESS_PATH/batch_unlearning_scaled_lr

# Paths for LM framework benchmark results
EVAL_FRAMEWORK_PATH=$BASE_PATH/eval_framework_tasks
BATCH_EVAL_PATH_FRAMEWORK=$EVAL_FRAMEWORK_PATH/batch_unlearning
SEQ_EVAL_PATH_FRAMEWORK=$EVAL_FRAMEWORK_PATH/sequential_unlearning
LLM_UNLEARN_EVAL_PATH_FRAMEWORK=$EVAL_FRAMEWORK_PATH/llm_unlearning_reproduced
BATCH_SCALED_LR_EVAL_PATH_FRAMEWORK=$EVAL_FRAMEWORK_PATH/batch_unlearning_scaled_lr

# Paths for saving results combined
EVAL_COMBINED_PATH=$BASE_PATH/eval_combined
BATCH_EVAL_COMBINED_PATH=$EVAL_COMBINED_PATH/batch_unlearning
SEQ_EVAL_COMBINED_PATH=$EVAL_COMBINED_PATH/sequential_unlearning
LLM_UNLEARN_EVAL_COMBINED_PATH=$EVAL_COMBINED_PATH/llm_unlearning_reproduced
BATCH_SCALED_LR_EVAL_COMBINED_PATH=$EVAL_COMBINED_PATH/batch_unlearning_scaled_lr

VENV_ACTIVATION_PATH=$BASE_PATH/../venv/bin/activate

# Ensure we are in the toplevel directory of the SNLP_GCW_data_analysis repo!
if [ `basename $BASE_PATH` != "SNLP_GCW_data_analysis" ]
then
    echo "Need to be in the SNLP_GCW_data_analysis directory to succesfully run this scrip!"
    exit 1
fi

# Make sure we can source the venv
if `source $VENV_ACTIVATION_PATH`
then
    echo "Succesfully sourced the python venv"
else
    echo "Make sure the path $VENV_ACTIVATION_PATH points to an exisiting python venv with required packages installed!"
    exit 1
fi


batch_runs=$(ls $BATCH_EVAL_PATH_HARMFULNESS)
seq_runs=$(ls $SEQ_EVAL_PATH_HARMFULNESS)
llm_unlearning_reproduced_runs=$(ls $LLM_UNLEARN_EVAL_PATH_HARMFULNESS)
batch_scaled_lr_runs=$(ls $BATCH_SCALED_LR_EVAL_PATH_HARMFULNESS)

echo "Processing results for batch runs.."
for run in ${batch_runs[@]}
do
    echo "Run: $run"
    mkdir -p $BATCH_EVAL_COMBINED_PATH/$run
    python3 eval_results_combined.py \
     --plot_title `basename $run` \
     --eval_csv_framework $BATCH_EVAL_PATH_FRAMEWORK/$run \
     --eval_csv_harmfulness $BATCH_EVAL_PATH_HARMFULNESS/$run \
     --log_dir $BATCH_EVAL_COMBINED_PATH/$run
done

echo "Processing results for sequential runs.."
for run in ${seq_runs[@]}
do
    echo "Run: $run"
    mkdir -p $SEQ_EVAL_COMBINED_PATH/$run
    python3 eval_results_combined.py \
     --plot_title `basename $run` \
     --eval_csv_framework $SEQ_EVAL_PATH_FRAMEWORK/$run \
     --eval_csv_harmfulness $SEQ_EVAL_PATH_HARMFULNESS/$run \
     --log_dir $SEQ_EVAL_COMBINED_PATH/$run
done

echo "Processing results for llm_unlearning_reproduced runs.."
for run in ${llm_unlearning_reproduced_runs[@]}
do
    echo "Run: $run"
    mkdir -p $LLM_UNLEARN_EVAL_COMBINED_PATH/$run
    python3 eval_results_combined.py \
     --plot_title `basename $run` \
     --eval_csv_framework $LLM_UNLEARN_EVAL_PATH_FRAMEWORK/$run \
     --eval_csv_harmfulness $LLM_UNLEARN_EVAL_PATH_HARMFULNESS/$run \
     --log_dir $LLM_UNLEARN_EVAL_COMBINED_PATH/$run
done
echo "Processing results for batch scaled lr runs.."
for run in ${batch_scaled_lr_runs[@]}
do
    echo "Run: $run"
    mkdir -p $BATCH_SCALED_LR_EVAL_COMBINED_PATH/$run
    python3 eval_results_combined.py \
     --plot_title `basename $run` \
     --eval_csv_framework $BATCH_SCALED_LR_EVAL_PATH_FRAMEWORK/$run \
     --eval_csv_harmfulness $BATCH_SCALED_LR_EVAL_PATH_HARMFULNESS/$run \
     --log_dir $BATCH_SCALED_LR_EVAL_COMBINED_PATH/$run
done
echo "Done"