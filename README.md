# Quantile-based Bias Initialization for Efficient Private Data Reconstruction in Federated Learning

## Requirements
We recommend setting up a dedicated environment with `Python 3.11` then
```bash
pip install -r requirements.txt
```

## Quick demo
For a quick demo including visualizations we provide two jupyter notebooks:
`Quick_Demo_Attack_(QBI).ipynb` and `Quick_Demo_Defense_(AGGP).ipynb`.

Run them via `jupyter notebook` or `jupyter lab` -> open notebook -> `Run all cells`.

(~3s runtime for both, running on CPU only, CIFAR-10 dataset will be downloaded automatically)
### ImageNet demo (Local dataset required)
For demos on the ImageNet dataset, we provide `Demo_Attack_ImageNet_(QBI).ipynb` and `Demo_Defense_ImageNet_(AGGP).ipynb`.
(~15s runtime each)

## Datasets
The CIFAR-10 and IMDB datasets will be downloaded automatically when an experiment is run, and saved in the `data`directory.
To  run experiments on ImageNet, the dataset has to be manually downloaded from [image-net.org](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and placed in the `data` directory.



## Running experiments
The files `exp_{DATASET}_{METHOD}_<Normalization>.py` contain the code to reproduce our results reported in Table 1-6, and can be run without any arguments.
They will produce a file called `results_{DATASET}_{METHOD}.csv` in the `results` directory.

E.g. use the following command to run the experiments on CIFAR-10 with QBI:
```bash
python exp_CIFAR-10_QBI.py
```

### Metrics
The following metrics (as defined in Section 4) are reported in the results files:
- **A**: Active neurons
- **P**: Extraction-Precision
- **R**: Extraction-Recall (*Percentage of perfectly reconstructed samples*)

95% confidence intervals across the 10 runs are also reported.


### Customizing experiments
The default experiments will run each (`layer_size`,`batch_size`) combination with 10 different random initializations.
Adjust the variables `runs_per_setting`, `layer_sizes` or `batch_sizes`, initialized in the main method of each experiment, to your liking.

## Supplementary
Contained in the `supplementary` directory are the following files: 
1. `conv2d_identity.py` Minimal working example for conv layers working as identity functions on multi-channel data, as described in Algorithm 3.
2. `overlay2-32.py` Script that produced Figure 5 (effect of averaging increasing number of samples)
3. `similarity_metrics.py` Script that produced Figure 6 (similarity metrics of increasing number of samples)