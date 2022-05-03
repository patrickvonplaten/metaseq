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

4. Download the small model as shown [here](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT).

```
wget https://dl.fbaipublicfiles.com/opt/v1_20220502/125m/reshard-model_part-0.pt
```

5. Try to load the model withe the following Python code:

```py
#!/usr/bin/env python3
from metaseq import checkpoint_utils

model = checkpoint_utils.load_model_ensemble_and_task(["./reshard-model_part-0.pt"], arg_overrides={"vocab_filename": "./vocab.json", "merges_filename": "./merges.txt"})
```

6. You'll probs see the following error coming from the Megatron-LM library:

```
No CUDA runtime is found, using CUDA_HOME='/usr'
Traceback (most recent call last):
  File "/home/patrick_huggingface_co/add_opt/./load_model.py", line 5, in <module>
    model = checkpoint_utils.load_model_ensemble_and_task(["./reshard-model_part-0.pt"], arg_overrides={"vocab_filename": "/home/patrick_huggingface_co/add_opt/vocab.json", "merges_filename": "/home/patrick_huggingface_co/add_opt/merges.txt"})
  File "/home/patrick_huggingface_co/metaseq/metaseq/checkpoint_utils.py", line 506, in load_model_ensemble_and_task
    model = task.build_model(cfg.model)
  File "/home/patrick_huggingface_co/metaseq/metaseq/tasks/base_task.py", line 560, in build_model
    model = models.build_model(args, self)
  File "/home/patrick_huggingface_co/metaseq/metaseq/models/__init__.py", line 89, in build_model
    return model.build_model(cfg, task)
  File "/home/patrick_huggingface_co/metaseq/metaseq/model_parallel/models/transformer_lm.py", line 47, in build_model
    embed_tokens = cls.build_embedding(
  File "/home/patrick_huggingface_co/metaseq/metaseq/model_parallel/models/transformer_lm.py", line 82, in build_embedding
    embed_tokens = VocabParallelEmbedding(
  File "/home/patrick_huggingface_co/Megatron-LM/megatron/mpu/layers.py", line 190, in __init__
    self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
  File "/home/patrick_huggingface_co/Megatron-LM/megatron/mpu/initialize.py", line 258, in get_tensor_model_parallel_world_size
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())
  File "/home/patrick_huggingface_co/Megatron-LM/megatron/mpu/initialize.py", line 215, in get_tensor_model_parallel_group
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
AssertionError: intra_layer_model parallel group is not initialized
```
