#! /bin/bash

BASE_PATH=`pwd`
EVAL_FRAMEWORK_PATH=$BASE_PATH/eval_framework_tasks
BATCH_EVAL_PATH=$EVAL_FRAMEWORK_PATH/batch_unlearning
SEQ_EVAL_PATH=$EVAL_FRAMEWORK_PATH/sequential_unlearning
LLM_UNLEARN_EVAL_PATH=$EVAL_FRAMEWORK_PATH/llm_unlearning_reproduced
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


batch_runs=$(ls $BATCH_EVAL_PATH)
seq_runs=$(ls $SEQ_EVAL_PATH)
llm_unlearning_reproduced_runs=$(ls $LLM_UNLEARN_EVAL_PATH)

echo "Processing results for batch runs.."
for run in ${batch_runs[@]}
do
    echo "Run: $run"
    python3 eval_framework.py \
     --plot_title `basename $run` \
     --log_dir $BATCH_EVAL_PATH/$run/eval_results
done

echo "Processing results for sequential runs.."
for run in ${seq_runs[@]}
do
    echo "Run: $run"
    python3 eval_framework.py \
     --plot_title `basename $run` \
     --log_dir $SEQ_EVAL_PATH/$run/eval_results
done

echo "Processing results for llm_unlearning_reproduced runs.."
for run in ${llm_unlearning_reproduced_runs[@]}
do
    echo "Run: $run"
    python3 eval_framework.py \
     --plot_title `basename $run` \
     --log_dir $LLM_UNLEARN_EVAL_PATH/$run/eval_results
done

echo "Done"
