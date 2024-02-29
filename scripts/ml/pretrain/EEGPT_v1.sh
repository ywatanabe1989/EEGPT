#!/bin/bash

# Define arrays of parameter values
# For consistency, even single values are defined as arrays
CONV_N_BLOCKS=(3)
CONV_SPLIT_SIZE_PTS=(32)
VQ_NUM_EMBEDDINGS=(16)
VQ_EMBEDDING_DIM=(16) # dependes on Conv Layer
VQ_INITIALIZER=(uniform)
VQ_WEIGHT_NORM=(False True)
VQ_SOFTMAX_TYPE=(gumbel_softmax_hard)
VQ_COMMITMENT_COEF=(1.0)
VQ_ENTROPY_COEF=(0.3)
VQ_DIVERSITY_COEF=(0 0.3)
VQ_PREDICTION_COEF=(1.0)
TF_HIDDEN_DIM=(16)
TF_DIM_FEEDFORWARD=(2048)
TF_N_HEADS=(2 4) # TF_HIDDEN_DIM must be divisible by TF_N_HEADS
TF_N_ENC_LAYERS=(2 6)
TF_N_DEC_LAYERS=(2 6)
TF_DROPOUT_RATIO=(0.1)

# Function to run the command with expanded parameters
run_command() {
    conv_n_blocks=$1
    conv_split_size_pts=$2
    vq_num_embeddings=$3
    vq_embedding_dim=$4
    vq_initializer=$5
    vq_weight_norm=$6
    vq_softmax_type=$7
    vq_commitment_coef=$8
    vq_entropy_coef=$9
    vq_diversity_coef=${10}
    vq_prediction_coef=${11}
    tf_hidden_dim=${12}
    tf_dim_feedforward=${13}
    tf_n_heads=${14}
    tf_n_enc_layers=${15}
    tf_n_dec_layers=${16}
    tf_dropout_ratio=${17}

    command="./scripts/ml/pretrain_EEGPT.py \
        --conv_n_blocks $conv_n_blocks \
        --conv_split_size_pts $conv_split_size_pts \
        --vq_num_embeddings $vq_num_embeddings \
        --vq_embedding_dim $vq_embedding_dim \
        --vq_initializer $vq_initializer \
        --vq_weight_norm $vq_weight_norm \
        --vq_softmax_type $vq_softmax_type \
        --vq_commitment_coef $vq_commitment_coef \
        --vq_entropy_coef $vq_entropy_coef \
        --vq_diversity_coef $vq_diversity_coef \
        --vq_prediction_coef $vq_prediction_coef \
        --tf_hidden_dim $tf_hidden_dim \
        --tf_dim_feedforward $tf_dim_feedforward \
        --tf_n_heads $tf_n_heads \
        --tf_n_enc_layers $tf_n_enc_layers \
        --tf_n_dec_layers $tf_n_dec_layers \
        --tf_dropout_ratio $tf_dropout_ratio"

    # Execute the command
    echo "$command" # For debugging
    eval "$command"
}

export -f run_command

# Use parallel to run the function with combinations of parameters
parallel --jobs 4 run_command ::: "${CONV_N_BLOCKS[@]}" ::: "${CONV_SPLIT_SIZE_PTS[@]}" ::: "${VQ_NUM_EMBEDDINGS[@]}" ::: "${VQ_EMBEDDING_DIM[@]}" ::: "${VQ_INITIALIZER[@]}" ::: "${VQ_WEIGHT_NORM[@]}" ::: "${VQ_SOFTMAX_TYPE[@]}" ::: "${VQ_COMMITMENT_COEF[@]}" ::: "${VQ_ENTROPY_COEF[@]}" ::: "${VQ_DIVERSITY_COEF[@]}" ::: "${VQ_PREDICTION_COEF[@]}" ::: "${TF_HIDDEN_DIM[@]}" ::: "${TF_DIM_FEEDFORWARD[@]}" ::: "${TF_N_HEADS[@]}" ::: "${TF_N_ENC_LAYERS[@]}" ::: "${TF_N_DEC_LAYERS[@]}" ::: "${TF_DROPOUT_RATIO[@]}"

# ./scripts/ml/pretrain_EEGPT.sh
