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
