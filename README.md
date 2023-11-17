# LRGNet - Low Rank Graph Network with Intrinsic Fact Dimension on Evidence Fact-Checking



## Data
All data available at: https://disk.pku.edu.cn:443/link/F593EBC724B9448AD47AA2BD790BB62D

### Notation
"_correct" for only integrate correction. "_lengthened" for lengthened data. "_modified" for category-balanced data.



## Code

### Notation
"FEVER_" for FEVER data processing. "CHEF_" for CHEF data processing.

"no_suffix" for original LLAMA. "_lr" for LRGAT-LLAMA.

### Requirement
peft        : 0.5.0

torch       : 1.13.1

loralib     : 0.1.2

transformers: 4.34.0.dev0

accelerate  : 0.23.0

datasets    : 2.10.1

### Model
Llama       : decapoda-research/llama-7b-hf

Llama2      : meta-llama/Llama-2-7b-hf



## TODO
Multi-device data parallel. 



## Attention
Do not try eval batch size > 1.
Do not use left padding for llama-2.
