# LRGNet - Low Rank Graph Network with Intrinsic Fact Dimension on Evidence Fact-Checking

CHEF data available at: https://disk.pku.edu.cn:443/link/F593EBC724B9448AD47AA2BD790BB62D

FEVER data available at: https://disk.pku.edu.cn:443/link/FC85BC31162E109FEBAA0920435D22E2


## Code

### Notation 
"Origin" for the latest stable code. "Trial" for the latest development code with readable results. "LM" for language modeling. "CLS" for sequence classification.


## TODO

(11.03.1) Device selection is insane in Trainer framework. You have to use exactly four GPUs to run these codes. Fix the modules that use ".cuda()" approach to select devices manually.

(11.03.2) Layer-specific <adj> tensor construction in graph attention networks.

(11.03.3) Using a training dataset (len=10) and a test dataset (len=10), batch size equals 2, and gradient accumulation step equals 2, the hook method has hooked 15 outputs and the total labels predicted are 54 (maybe 2*2*2*3 in training and 3*10 in evaluate). Find out the tracking mechanism of the hook, how to track the evaluate steps or how the evaluate steps are tracked, and the implementation.
