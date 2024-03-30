#! /bin/bash

BASE_PATH=`pwd`
EVAL_HARMFULNESS_PATH=$BASE_PATH/eval_harmfulness
BATCH_EVAL_PATH=$EVAL_HARMFULNESS_PATH/batch_unlearning
SEQ_EVAL_PATH=$EVAL_HARMFULNESS_PATH/sequential_unlearning
LLM_UNLEARN_EVAL_PATH=$EVAL_HARMFULNESS_PATH/llm_unlearning_reproduced
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
for batch_run in ${batch_runs[@]}
do
    echo "Run: $batch_run"
    python3 eval_harmfulness.py \
     --plot_title `basename $batch_run` \
     --eval_dataset $BATCH_EVAL_PATH/$batch_run \
     --output_dir $BATCH_EVAL_PATH/$batch_run
done

echo "Processing results for sequential runs.."
for seq_run in ${seq_runs[@]}
do
    echo "Run: $seq_run"
    python3 eval_harmfulness.py \
     --plot_title `basename $seq_run` \
     --eval_dataset $SEQ_EVAL_PATH/$seq_run \
     --output_dir $SEQ_EVAL_PATH/$seq_run
done

echo "Processing results for llm_unlearning_reproduced runs.."
for unlearn_run in ${llm_unlearning_reproduced_runs[@]}
do
    echo "Run: $unlearn_run"
    python3 eval_harmfulness.py \
     --plot_title `basename $unlearn_run` \
     --eval_dataset $LLM_UNLEARN_EVAL_PATH/$unlearn_run \
     --output_dir $LLM_UNLEARN_EVAL_PATH/$unlearn_run
done
echo "Done"