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
## How to run this repository (new)
```src/main.py``` is the entry script. Use below command to get information on different flags.   

```python src/main.py --help```

```commandline
Output:

usage: main.py [-h] [--output_tag OUTPUT_TAG] [--eval] [--num_epochs NUM_EPOCHS] dataset model

positional arguments:
  dataset               Dataset type. Should be from this list: original, original_permuted, combinatorial, combinatorial_permuted, dirichlet, dirichlet_permuted
  model                 Model Type. Should be from this list: partially_invariant_nn, hartford

optional arguments:
  -h, --help            show this help message and exit
  --output_tag OUTPUT_TAG
                        Output tag used to save results or fetch saved results. Necessary if "--eval" flag is used (default: None)
  --eval                If specified, returns validation accuracy of saved model under "--output_tag" (default: False)
  --num_epochs NUM_EPOCHS
                        Number of epochs for training (default: 20)


```
### Training

```python src/main.py 'original' 'invariantmlp' --output_tag='invariantmlp_original' --num_epochs=20```

(for me worked: ```python -m src.main.py 'original' 'invariantmlp' --output_tag='invariantmlp_original' --num_epochs=20```)
for xgboost: 


### Evaluate

```python src/main.py 'original' 'invariantmlp' --output_tag='invarinatmlp_original' --eval```

For XGBoost:  ```python src/main.py 'dirichlet' 'xgboost' --output_tag='small_dirichlet' --eval```

### Tensorboard Visualization

Tensorborad visualizations are supported for training/validation loss and accuracy. All the runs are stored under ```data/runs/tensorboard```. Use below command to launch tensorboard.  

```tensorboard --logdir=data/runs/tensorboard --port=9009```

## How to run this repository

1. Unpack the file v26.gz in the same folder
2. Run train_demo.py. This should generate the following output:

```
Epoch 1/20
1091/1091 [==============================] - 2s 1ms/step - loss: 2.2385 - accuracy: 0.1769 - val_loss: 1.7860 - val_accuracy: 0.2596
Epoch 2/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.6828 - accuracy: 0.2987 - val_loss: 1.6647 - val_accuracy: 0.3042
Epoch 3/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.5747 - accuracy: 0.3334 - val_loss: 1.6231 - val_accuracy: 0.3214
Epoch 4/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.5049 - accuracy: 0.3626 - val_loss: 1.6255 - val_accuracy: 0.3167
Epoch 5/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.4721 - accuracy: 0.3669 - val_loss: 1.6057 - val_accuracy: 0.3260
Epoch 6/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.4412 - accuracy: 0.3889 - val_loss: 1.5877 - val_accuracy: 0.3367
Epoch 7/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.4064 - accuracy: 0.3980 - val_loss: 1.6235 - val_accuracy: 0.3287
Epoch 8/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.3791 - accuracy: 0.4078 - val_loss: 1.5937 - val_accuracy: 0.3385
Epoch 9/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.3619 - accuracy: 0.4098 - val_loss: 1.5893 - val_accuracy: 0.3370
Epoch 10/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.3426 - accuracy: 0.4217 - val_loss: 1.5819 - val_accuracy: 0.3421
Epoch 11/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.3195 - accuracy: 0.4320 - val_loss: 1.5848 - val_accuracy: 0.3413
Epoch 12/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.3081 - accuracy: 0.4433 - val_loss: 1.5928 - val_accuracy: 0.3432
Epoch 13/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.2821 - accuracy: 0.4508 - val_loss: 1.5896 - val_accuracy: 0.3464
Epoch 14/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.2686 - accuracy: 0.4560 - val_loss: 1.6062 - val_accuracy: 0.3467
Epoch 15/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.2661 - accuracy: 0.4531 - val_loss: 1.6193 - val_accuracy: 0.3494
Epoch 16/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.2504 - accuracy: 0.4588 - val_loss: 1.6206 - val_accuracy: 0.3462
Epoch 17/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.2289 - accuracy: 0.4721 - val_loss: 1.6167 - val_accuracy: 0.3500
Epoch 18/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.2130 - accuracy: 0.4772 - val_loss: 1.6467 - val_accuracy: 0.3480
Epoch 19/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.1963 - accuracy: 0.4884 - val_loss: 1.6493 - val_accuracy: 0.3492
Epoch 20/20
1091/1091 [==============================] - 1s 1ms/step - loss: 1.1878 - accuracy: 0.4887 - val_loss: 1.6420 - val_accuracy: 0.3525
```

## Dirichlet projections
The folder fundamental_domain_projections contains an implementation of an approximation of the Dirichlet fundamental domain projection, which is explained in section E.2 in the preprint. The important function is dirichlet_ord in file is discrete_gradient_ascent.py. Here, x is the input matrix, x0 the matrix specifying the order. You also need to specify a generating set for the symmetry group. Some of them have been implemented in the file generating_sets.

## 8 Feb 2023 homework assignment

```
data
> raw
> > v26.gz
> projections
> > 6 files
> runs
> > tensorboard files will go here
> saved_models
src
> dataset.py -> creates dataset - done
> main.py -> manages training - Pragya
> models
> > hartford.py - Daniel
> > xgboost.py - Peter
> > partially_invariant.py - Pragya
> > vanilla_nn.py - Peter
> > vision_transformer.py - Peter
config
> constants.py
util
> generating_sets.py
> ... (as before)
training_suite.ipynb - Peter
```