# Data Synthesis via Differentially Private Markov Random Fields

Kuntai Cai, Xiaoyu Lei, Jianxin Wei, Xiaokui Xiao

caikt@comp.nus.edu.sg

This project provides the implementation of "Data Synthesis via Differentially Private Markov Random Fields". You can generate a synthetic dataset with PrivMF and reproduce the experimental results using a single command.

Part of this repository reuses the code of Ryan McKenna et al. Thanks for their great work.

Ryan McKenna (2019.7) Graphical-model based estimation and inference for differential privacy [Source code]. https://github.com/ryan112358/private-pgm

## Get Started

These codes require Python3 and need to run on a GPU that supports `cupy`. The dependencies can be installed via

    $ pip3 install -r requirements.txt

After that, you can simply reproduce the experimental results reported in the paper via

    $ python3 script.py

By default, it runs PrivMRF once on each of the four datasets and five values of $\epsilon$ and reports the total variation distances (TVD). It takes two to five hours to complete, depending on the performance of the computer. You may modify the codes in script.py to run PrivMRF only in some cases.

You can also run PrivMRF only once with a specified setting via

    $ python3 main.py

This will generate a synthetic dataset but it does not report any result. It first preprocesses the dataset and runs PrivMRF via

    main('nltcs', 'test', 0.8, task='TVD')

`nltcs` specified the dataset. We support `nltcs, acs, adult, br2000` currently. `test` is an arbitrary identifier that specifies the name of the output data. `0.8` specified the value of $\epsilon$. It could be `0.1, 0.2, 0.4, 0.8, 1.6, 3.2`. Note that we set $\delta=10^{-5}$ by default. `task` specified the aim of generating data. If you set `task='SVM'`, it will only use 80% of the data to generating synthetic data since the other 20% are used for testing SVM classifiers.

## Reproduce the SVM results

For simplicity, we comment out the codes that run SVM experiments in script.py. You can uncomment them to reproduce SVM experiments.

    # run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25)
    # run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25, generate=False)

The first line generates synthetic data. The second line trains SVM classifiers on the synthetic data and tests their mis-classification rates, which is a parallel program. Each combination of a dataset, method, and epsilon needs one process. You may want to avoid calling too many processes by setting `data_list, method_list, epsilon_list` such that their sizes are small. Training and testing SVMs for one dataset and one $\epsilon$ take 1 to 6 hours.

## Run PrivMRF on other datasets

`./data` contains all the datasets used in the experiments. The data should be in csv format and its first line needs to be the names of attributes (columns). Suppose that the name of your dataset is `some_data`. Its content should be in `some_data.csv` and you will need `some_data_hierarchy.json` to specifies the attribute hierarchy. In the `adult` dataset, for each attribute, we merge values into bins. Since the number of records in a bin is larger than the number of records of each value of the bin, merging values provides more resistance to noise. For the `adult` dataset, PrivMRF plus this technique provides slightly better performance. PrivBayes[1] provides a detailed description of attribute hierarchy. For Makov random field, we apply attribute hierarchy to marginal distributions. That is, we only count the number of records in each bin instead of counting the number of records of each possible combination of values.

Note that attribute hierarchy does not always work for PrivMRF. You can simply ignore this technique and just send a one-level hierarchy to PrivMRF. In this case, leave the value of `node_list` empty, enumerate all the possible values in `value_list`, set `level_to_size` the size of the domain. Refer to `./data/acs_hierarchy.json` for example.

## Calculating privacy budget

The privacy budgets are calculated via `cal_privacy_budget()` in tools.py. You can call this function to calculate the privacy budgets of different $\epsilon$ and $\delta$.