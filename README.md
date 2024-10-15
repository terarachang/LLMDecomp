## When Parts Are Greater Than Sums: Individual LLM Components Can Outperform Full Models (EMNLP 2024)
> Ting-Yun Chang, Jesse Thomason, and Robin Jia<br>

> **Paper**: https://arxiv.org/abs/2406.13131
> **Blog:** https://terarachang.github.io/projects/llm-decomp.html

## Methods
### Quick Start
```
export HF_TOKEN="YOUR TOKEN"
pip install -r requirements.txt
```

### Component Reweighting
``` bash
$ bash scripts/comp_rw.sh
```
- Implementation of model decomposition: [decompose.py](decompose.py)
- Implementation of reweighting: [train_components.py](train_components.py)


### Standard ICL
``` bash
$ bash scripts/standard.sh
```

### Calib+
``` bash
$ bash scripts/calibration.sh
```
- Implementation of trainable calibration: [train_calib.py](train_calib.py)

### Adding New Models
- Our repo supports LLMs in the Llama and Mistral family
- To support new models, please add hooks to the model and follow the naming convention of [my_modeling_llama.py](my_modeling_llama.py)
- If the new model also uses RMSNorm, the [decompose.py](decompose.py) file is directly applicable. Otherwise, please take care of layernorms, which may greatly influence model performance!
- *We do not fully adopt [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to avoid numerical issues in Llama-3 and reduce computation overhead
