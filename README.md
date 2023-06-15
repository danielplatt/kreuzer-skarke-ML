# kreuzer-skarke-invariant-ML

## The dataset

The file v26.gz is taken from http://quark.itp.tuwien.ac.at/~kreuzer/V. It contains around 78,000 reflexive polytopes, each corresponding to a different Calabi-Yau 3-fold.
Entries look like this:

```
4 26  M:51 26 N:11 10 H:6,46 [-80]
   1   0   0   0   0  -1   0   0   0   0  -2   2  -1   1   0  -1   0  -2   2   1  -1  -2   0   2   0   1
   0   1   0   0  -1   0   2   0   0  -2   0   0   1  -1  -2   0  -1   0   0   0  -2  -1   1   1   2   2
   0   0   1   0   1   1  -1   1  -1   1   1  -1   1   0   1  -1   1   0  -1   0   1   1  -2  -2  -2  -2
   0   0   0   1   1   1  -1  -1   1   1   1  -1   0   1   0   1  -1   1  -2  -2   1   1   0  -2  -1  -2
```

The numbers in the first line are explained in the following (taken from http://www2.macaulay2.com/Macaulay2/doc/Macaulay2-1.18/share/doc/Macaulay2/ReflexivePolytopesDB/html/___Kreuzer-__Skarke_spdescription_spheaders.html).
It follows an explanation for the header `4 10  M:25 10 N:10 9 H:5,20 [-30]`. (That is: same format, different numbers.)
            
'4 10': the first 2 numbers are the number of rows and columns of the matrix $A$            
            
'M:25 10': number of lattice points and the number of vertices of the 4-dimensional lattice polytope $P$ which is the convex hull of the columns of the matrix $A$            
            
'N: 10 9' is the number of lattice points and the number of vertices of the polar dual polytope $P^o$ of $P$            
            
'H: 5,20 [-30]' are the Hodge numbers $h^{1,1}(X)$, $h^{1,2}(X)$, and the topological Euler characteristic of $X$, where $X$ is the Calabi-Yau variety described next            
The last four lines stand for 26 vectors in 4-dimensional Euclidean space.
Note that permuting columns and permuting rows of this matrix describes a polytope that encodes an isomorphic Calabi-Yau 3-fold.

Remark:
some entries are malformed, such as the following, which is copied verbatim from the raw data file.

```26 4  M:28 26 N:30 26 H:24,22 [4]
1 0 0 0 
0 1 0 0 
0 0 1 0 
0 0 0 1 
0 -1 1 1 
0 1 0 -1 
0 -1 0 1 
0 1 -1 -1 
0 0 0 -1 
0 0 -1 0 
0 -1 0 0 
0 0 1 1 
-1 0 0 1 
-1 -1 1 1 
-1 1 -1 -1 
-1 0 0 -1 
-1 -1 0 1 
-1 0 -1 0 
-1 -1 0 0 
-1 1 0 -1 
-1 0 -1 -1 
-1 1 0 0 
-1 0 1 0 
-1 1 -1 0 
-1 -1 1 0 
-1 0 1 1 
```

### How to load pre-processed datasets

For our experiments, we used four datasets:

`original`: without any permutation or projection

`original_permuted`: random permutation applied, no projection 

`dirichlet`: no permutation, dirichlet projection applied

`dirichlet_permuted`: random permutation and dirichlet projection applied

Use below command to load above datasets in python:
```commandline
from src.dataset import *

dataset_name = 'original'
dataset = KreuzerSkarkeDataset(load_projections=True, projections_file=dataset_name)
```

## How to run this repository
1. Install dependencies using below command:

```pip install -r requirements.txt```

2. ```src/main.py``` is the entry script. Use below command to get information on different flags and how to run it:   

    
```python -m src.main.py --help```

Output:

```commandline
usage: -m [-h] [--output_tag OUTPUT_TAG] [--eval] [--num_epochs NUM_EPOCHS] dataset model

positional arguments:
  dataset               Dataset type. Should be from this list: original, original_permuted, combinatorial, combinatorial_permuted, dirichlet, dirichlet_permuted
  model                 Model Type. Should be from this list: invariantmlp, hartford, xgboost, vanilla_nn, vision_transformer, pointnet

options:
  -h, --help            show this help message and exit
  --output_tag OUTPUT_TAG
                        Output tag used to save results or fetch saved results. Necessary if "--eval" flag is used (default: None)
  --eval                If specified, returns validation accuracy of saved model under "--output_tag" (default: False)
  --num_epochs NUM_EPOCHS
                        Number of epochs for training (default: 20)
```
### Training

```python -m src.main.py <dataset> <model> --output_tag=<output_tag> --num_epochs=20```

`<dataset>` is the dataset used, selected from [ 'original', 'original_permuted', 'dirichlet', 'dirichlet_permuted' ]

`<model>` is the model used, selected from [ 'invariantmlp', 'hartford', 'xgboost', 'vanilla_nn', 'vision_transformer', 'pointnet' ] 

`<output_tag>` is the name used to save model output. 

Examples:

For invariant mlp: ```python -m src.main.py 'original' 'invariantmlp' --output_tag='invariantmlp_original' --num_epochs=20```)

For xgboost: ```python -m src.main 'original' 'xgboost' --output_tag='xgboost_original' --num_epochs=20```

For vision transformer: ```python -m src.main 'original' 'vision_transformer' --output_tag='vision_original' --num_epochs=1```

For vanilla cnn: ```python -m src.main 'original' 'vanilla_nn' --output_tag='vanilla_nn_original' --num_epochs=20```

### Evaluate

```python src/main.py 'original' 'invariantmlp' --output_tag='invarinatmlp_original' --eval```

For XGBoost:  ```python src/main.py 'dirichlet' 'xgboost' --output_tag='small_dirichlet' --eval```


### Tensorboard Visualization

Tensorborad visualizations are supported for training/validation loss and accuracy. All the runs are stored under ```data/runs/tensorboard```. 

Use below command to launch tensorboard:  

```tensorboard --logdir=data/runs/tensorboard --port=9009```
