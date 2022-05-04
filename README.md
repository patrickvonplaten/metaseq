# How to load the small model

1. Clone this repo and install all dependencies:

```
git clone https://github.com/patrickvonplaten/metaseq.git
cd metaseq
pip3 install -e .
```

2. Install Megatron LM as described in the [official setup.md](https://github.com/facebookresearch/metaseq/blob/main/docs/setup.md).

```
git clone --branch fairseq_v2 https://github.com/ngoyal2707/Megatron-LM.git
cd Megatron-LM
pip3 install six regex
pip3 install -e .
```

3. Create a directory where you save the model and tokenizer
```
mkdir -p add_opt
cd add_opt
```

Now load the HF GPT2 tokenizer and save it:

```py
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("./")
```

4. Download the small model as shown [here](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT).

```
wget https://dl.fbaipublicfiles.com/opt/v1_20220502/350m/reshard.pt
```

5. Try to load the model withe the following Python code:

```py
import os

from megatron import get_args
from megatron.initialize import initialize_megatron
from metaseq import checkpoint_utils

path = "/home/patrick/add_opt"

# arguments taken from: https://arxiv.org/pdf/2205.01068.pdf | table 1
initialize_megatron(args_defaults={
    "micro_batch_size": 1, 
    "num_layers": 24, 
    "hidden_size": 1024, 
    "num_attention_heads": 16,
    "max_position_embeddings": 2048, # TODO check if it is the correct args
    "encoder_seq_length": 2048 # TODO check if it is the correct args
})

model = checkpoint_utils.load_model_ensemble_and_task(
#    [os.path.join(path, "reshard-model_part-0.pt"), os.path.join(path, "reshard-model_part-1.pt")],
    [os.path.join(path, "reshard.pt")],
    arg_overrides={
        "vocab_filename": os.path.join(path, "vocab.json"),
        "merges_filename": os.path.join(path, "merges.txt"),
    }
)
import ipdb; ipdb.set_trace()
```
