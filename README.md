# NNBellman

## Description
This project addressed the need for a faster method of generating agent decisions for an agent-based model. The original policy iteration method, while sufficient for a small number of agents was computationally cost prohibitive at the population scales demanded by the ABM. Using scripts documented in this repository, an exhaustive search for suitable architecture and hyperparameters was conducted to train a feedforward multilayer perceptron (MLP) as a replacement for the original iterative method. All data used to train the model and the models and metrics resulting from the hyperparameter grid search are available at [doi:10.5281/zenodo.14987728](https://doi.org/10.5281/zenodo.14987728)
The code structure for model training was based on the [PyTorch Template Project](https://github.com/victoresque/pytorch-template/).


## Installation
```bash
git clone https://github.com/vmgaribay/nnbellman.git
conda env create -f environment.yml
```

## Usage Notes 
The code in this repository was admitedly not designed for general reuse but the main files of interest to others may be...


- [Iterative_vs_NN.py](/Iterative_vs_NN.py) - Used to obtain the performance comparison values (set device to cpu if needed)
- [GenerateBellmanSample.ipynb](/GenerateBellmanSample.ipynb) - Used to generate the training and testing data
- [model.py](/model/model.py) - Experimental neural network achitectures 
- [as_both_grid_search_S21gpu.sh](BatchRuns/GridSearch/as_both_grid_search_S21gpu.sh) - Example of script used to conduct architecture/hyperparameter grid search
- [nn_performance_comparison.ipynb](nn_performance_comparison.ipynb) - Used to extract best model performance from log files

If cuda is unavailable on your machine, the code may still be run on cpu.

## Performance Comparison
The comparison of performance<sup>1</sup> for the original iterative method and the neural network equation mapping with chosen architecture and hyperparameters is listed in the following tables for n agents. The values and numbers in parenthesis respectively represent the mean and standard deviation<sup>2</sup> of 10 randomly seeded runs.
### n=1
| Metric | Iterative | MLP |
|--------|-----------|-----|
| Execution Time (s) | 34.80 (5.30) | 0.16 (0.15) |
| CPU Time (s) | 34.70 (5.28) | 0.12 (0.03) |
| CPU Memory Usage (MB) | 0.05 (0.01) | 18.56 (5.31) |
| GPU Time (s) | 34.8 (5.30) | 0.16 (0.15) |
| GPU Memory Usage (MB) | 7.67 (2.69) | 91.69 (2.69) |

### n=10
| Metric | Iterative | MLP |
|--------|-----------|-----|
| Execution Time (s) | 361.77 (29.53) | 0.12 (0.01) |
| CPU Time (s) | 360.77 (29.46) | 0.12 (0.00) |
| CPU Memory Usage (MB) | 0.08 (0.00) | 16.88 (0.00) |
| GPU Time (s) | 361.77 (29.53) | 0.12 (0.01) |
| GPU Memory Usage (MB) | 8.52 (0.00) | 92.55 (0.00) |

### n=100
| Metric | Iterative | MLP |
|--------|-----------|-----|
| Execution Time (s) | 3831.86 (347.49) | 0.18 (0.06) |
| CPU Time (s) | 3821.24 (346.52) | 0.13 (0.00) |
| CPU Memory Usage (MB) | 0.23 (0.00) | 16.88 (0.00) |
| GPU Time (s) | 3831.87 (347.49) | 0.18 (0.06) |
| GPU Memory Usage (MB) | 8.52 (0.00) | 92.55 (0.00) |

### n=1000
| Metric | Iterative | MLP |
|--------|-----------|-----|
| Execution Time (s) | 38957.39 (955.84) | 0.53 (0.23) |
| CPU Time (s) | 38852.38 (952.87) | 0.19 (0.03) |
| CPU Memory Usage (MB) | 3.01 (2.13) | 30.32 (7.08) |
| GPU Time (s) | 38957.58 (955.93) | 0.53 (0.23) |
| GPU Memory Usage (MB) | 1.72 (3.59) | 85.75 (3.59) |


<sup>1</sup> Run on [Snellius gcn partition](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660208/Snellius+hardware), established Q4 2022\
Hardware Information: 
Lenovo ThinkSystem SD650-N v2\
&nbsp;&nbsp;&nbsp;&nbsp;OS: Red Hat Enterprise Linux 9.4 (Plow)\
&nbsp;&nbsp;&nbsp;&nbsp;CPU: Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CPU family:           6\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model:                106\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Thread(s) per core:   1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Core(s) per socket:   36\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Socket(s):            2\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Stepping:             6\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CPU(s) scaling MHz:   100%\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CPU max MHz:          2400.0000\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CPU min MHz:          800.0000\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DRAM GiB per core:    7.111

&nbsp;&nbsp;&nbsp;&nbsp;GPU: NVIDIA A100-SXM4-40GB\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Driver Version:       565.57.01\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CUDA Version:         12.7\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Power Cap W:          400

<sup>2</sup> Caveat: Small/No variation on memory usage metrics may be sign of improper reset between runs; values should still be representative of peak useage.


## Context
For more information on the dataset generation and training process please refer to the main manuscript, doi: pending
```