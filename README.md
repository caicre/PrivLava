# PrivLava: Synthesizing Relational Data with Foreign Keys under Differential Privacy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Kuntai Cai, Xiaokui Xiao, Graham Cormode

caikt@comp.nus.edu.sg

This project provides an implementation of "PrivLava: Synthesizing Relational Data with Foreign Keys under Differential Privacy", which can generate a synthetic version of relational data with foreign keys under differential privacy.

## 1. Get Started

These codes require Python3.8 and need to run on a GPU that supports `cupy`. CUDA Toolkit 10.2 can be installed [here](https://developer.nvidia.com/cuda-10.2-download-archive).

The python dependencies can be installed via

```
pip3 install -r requirements.txt
```

After installation, you may run PrivLava via

```
python3 main_California.py exp0 3.20 California
```

*   ```main_California.py``` is a script for synthesizing a 2-table database.
*   ```exp0``` is an arbitrary identifier that specifies the name of the output data.
*   ```3.20``` is the value of $\epsilon$ in $(\epsilon,\delta)$-differential privacy. The value of $\delta$ is set to be $1/\mathrm{PersonNum}$ in line 114 in ```main_California.py```.

Then, you can evaluate the relative error via

```
python3 evaluate_2table_2individual_relative.py ./data/California/individual.csv ./data/California/individual_domain.json ./data/California/household.csv ./data/California/household_domain.json ./temp/exp0_California_3.20_individual.csv ./temp/exp0_California_3.20_household.csv 0.01
```


## 2. Dataset

We store the California dataset in ```./data/California```.

*   ```household.csv``` and ```individual.csv``` are tables, and the last column of ```individual.csv``` should be the foriegn key referencing ```household.csv```. Their first columns should be their primary key respectively.
*   ```household_domain.json``` and ```individual_domain.json``` provide the domain information of the attributes. Each contains a dict storing the types and domain sizes of the attributes. The ```type``` of attributes can be ```continuous``` or ```discrete``` while PrivLava will treat them all as discrete (categorical) attributes. ```evaluate_2table_relative.py``` generates random continuous domain for continuous attribute but random discontinuous domain for discrete attributes.
*   The domains of the attribute of the 2 tables should be in $\{0, 1, \cdots, \text{size}-1\}$, where sizes are given in the json domain file.

For performance,
*   We note that you may need a sufficiently large dataset to get a reasonable performance using PrivLava. For example, our California dataset has $1.69\times 10^6$ persons. To ensure that you have enough budget for synthesis, check the "latent size:" in the log of PrivLava, which should be at least 15 for reasonable performance.
*   You may also want to avoid a large attribute domain (say, larger than 50) by decomposing the large attribute to several small attributes, as a large domain may cause the marginals to be excessively noisy.
