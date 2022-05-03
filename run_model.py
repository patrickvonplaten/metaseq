import os

from megatron import get_args
from megatron.initialize import initialize_megatron
from metaseq import checkpoint_utils

path_model = os.path.join("/home/younes/Desktop/Work/metaseq-conversion", "models")
path_files = os.path.join("/home/younes/Desktop/Work/metaseq-conversion", "add_opt")

# arguments taken from: https://arxiv.org/pdf/2205.01068.pdf | table 1
initialize_megatron(args_defaults={
    "micro_batch_size":1, 
    "num_layers":12, 
    "hidden_size":768, 
    "num_attention_heads":12,
    "max_position_embeddings":2048, # TODO check if it is the correct args
    "encoder_seq_length":2048 # TODO check if it is the correct args
})

model = checkpoint_utils.load_model_ensemble_and_task(
    [os.path.join(path_model, "reshard-model_part-0.pt"), os.path.join(path_model, "reshard-model_part-1.pt")], 
        arg_overrides={
            "vocab_filename": 
            os.path.join(path_files, "vocab.json"), 
            "merges_filename": 
            os.path.join(path_files, "merges.txt"),
        }
    )