# LLM-Microscope

This repository contains the official implementation of the code for papers "LLM-Microscope: A Toolkit for Quantifying and Visualizing
Language Model Internals" and "[Your Transformer is Secretly Linear](https://arxiv.org/abs/2405.12250)".

![Linearity Profiles](linearity_profiles.png)

We've also created a pip package containing the functions from [demo notebook](https://github.com/AIRI-Institute/LLM-Microscope/blob/main/LLM_microscope.ipynb).

Use ```pip install llm-microscope``` to install it.

### Example (anisotropy, intrinsic dimension and linearity score)

```python
import torch
from llm_microscope import  (
  calculate_anisotropy_torch,
  intrinsic_dimension,
  procrustes_similarity,
  procrustes_similarity_centered,
  load_enwiki_text
)

device = 'cpu'

X = torch.randn((1000, 10)) # pseudo-random "features", 1000 vectors with dim=10.
Y = torch.randn((1000, 10)) # pseudo-random "features", 1000 vectors with dim=10.

anisotropy = calculate_anisotropy_torch(X) # anisotropy score
int_dim = intrinsic_dimension(X, device) # intrinsic dimension
linearity_score = procrustes_similarity(X, Y) # linearity score from the paper
centered_linearity_score = procrustes_similarity_centered(X, Y) # the same as linearity between X and Y - X


# You can also download the dataset that we used in the paper using load_enwiki_text function:
text = llm_microscope.load_enwiki_text()
```

### Example (Logit Lens)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_microscope import logit_lens, normalize_weights, plot_word_table, replace_bad_chars

device = 'cuda'
model_name = "facebook/opt-1.3b"
text = "Lorem Ipsum is simply dummy text of the printing"

tokenizer= AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).bfloat16().to(device)

tokens = tokenizer.encode(text)
words = [tokenizer.decode([tok]) for tok in tokens]
words = [replace_bad_chars(word) for word in words]

predictions, losses, decoded_words = logit_lens(model, tokenizer, text)
losses = normalize_weights(-losses, normalization_type="global") 

plot_word_table(decoded_words, losses, words)
```
