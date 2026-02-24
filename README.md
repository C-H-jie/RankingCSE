# RaCol

This repository contains the code for the paper **RaCol: Ranking-based Contrastive Learning for Sentence Representation**.

## Run

Run the following command in this directory:

```bash
python train.py
```

Please configure training parameters directly in the `main` function in `train.py`.

To apply our loss function on SumCSE and SynCSE, please modify the`data_args.train_file` parameter in `train.py` accordingly.

## Significance Test

The significance test script is in`t-test.py`.

The following are the significance test results of our model compared with other methods:

### Comparison with RankCSE

```plain text
Task                N      Ours    RankCSE     Delta       p(one)    Better     Result
----------------------------------------------------------------------------------------
STSBenchmark     1116   85.431%   81.528%   +3.903%    0.000e+00      1000       PASS
STS12            3108   76.627%   74.116%   +2.511%    2.000e-02       980       PASS
STS13            1500   85.594%   85.768%   -0.174%    5.960e-01       404  FAIL(Direction reversed)
STS14            3750   79.445%   78.320%   +1.126%    1.460e-01       854  FAIL(Not significant)
STS15            3000   85.936%   84.616%   +1.320%    4.900e-02       951       PASS
STS16            1186   83.650%   80.864%   +2.786%    0.000e+00      1000       PASS
----------------------------------------------------------------------------------------
PASS 4/6 
```

#### Macro Average Spearman Significance (Cross-Task)

Definition: Calculate Spearman correlation for each task first, then compute the task-level macro average.

```plain text
ours macro Spearman:   82.7805%
simcse macro Spearman: 80.8684%
macro Delta (ours-sim): +1.9122%
bootstrap p(one-tailed): 0.000e+00
95.0% CI of Delta: [1.3841%, 2.4914%]
Result: PASS
```

### Comparison with SynCSE

```plain text
Task                N      Ours    SynCSE     Delta       p(one)    Better     Result
----------------------------------------------------------------------------------------
STSBenchmark     1116   85.431%   84.269%   +1.163%    3.500e-02       965       PASS
STS12            3108   76.627%   76.145%   +0.482%    2.470e-01       753  FAIL(Not significant)
STS13            1500   85.594%   84.410%   +1.183%    5.000e-03       995       PASS
STS14            3750   79.445%   79.231%   +0.214%    3.670e-01       633  FAIL(Not significant)
STS15            3000   85.936%   84.846%   +1.090%    2.000e-02       980       PASS
STS16            1186   83.650%   82.877%   +0.773%    1.220e-01       878  FAIL(Not significant)
----------------------------------------------------------------------------------------
PASS 3/6 
```

#### Macro Average Spearman Significance (Cross-Task)

Definition: Calculate Spearman correlation for each task first, then compute the task-level macro average.

```plain text
ours macro Spearman:   82.7805%
simcse macro Spearman: 81.9631%
macro Delta (ours-sim): +0.8174%
bootstrap p(one-tailed): 0.000e+00
95.0% CI of Delta: [0.4581%, 1.1749%]
Result: PASS
```

### Comparison with SimCSE

```plain text
Task                N      Ours    SimCSE     Delta       p(one)    Better     Result
----------------------------------------------------------------------------------------
STSBenchmark     1116   85.431%   80.151%   +5.280%    0.000e+00      1000       PASS
STS12            3108   76.627%   69.093%   +7.534%    0.000e+00      1000       PASS
STS13            1500   85.594%   81.441%   +4.153%    0.000e+00      1000       PASS
STS14            3750   79.445%   72.661%   +6.784%    0.000e+00      1000       PASS
STS15            3000   85.936%   81.498%   +4.438%    0.000e+00      1000       PASS
STS16            1186   83.650%   79.832%   +3.818%    0.000e+00      1000       PASS
----------------------------------------------------------------------------------------
PASS 6/6 
```

#### Macro Average Spearman Significance (Cross-Task)

Definition: Calculate Spearman correlation for each task first, then compute the task-level macro average.

```plain text
ours macro Spearman:   82.7805%
simcse macro Spearman: 77.4461%
macro Delta (ours-sim): +5.3345%
bootstrap p(one-tailed): 0.000e+00
95.0% CI of Delta: [4.7859%, 5.8900%]
Result: PASS
```
