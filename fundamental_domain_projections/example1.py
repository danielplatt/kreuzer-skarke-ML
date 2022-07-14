import os, sys, json, numpy as np, pandas as pd, pickle, itertools
from sklearn.preprocessing import MinMaxScaler
from typing import TypeVar, Union, List, Literal, Tuple
from fundamental_domain_projections.dirichlet.dirichlet_dataset import *


def create_perturbation(
    X:np.ndarray, 
    kind:str='guassian', 
    feature_range:Tuple[float, float]=(-0.01, 0.01), 
    scale:float=1
) -> np.ndarray:
    '''
    Arguments:
    ----------
        X (np.ndarray): Input data matrix
        
        kind (str): Defaults to `'guassian'`. The kind of noise to generate.
        
        feature_range (Tuple[float, float]): Feature range to limit the noise to.
            Defaults to `(-0.01, 0.01)`.
        
        scale (float): Defaults  to `1`. A post min-max re-scaling via simple 
            multiplication to give further control over the noise.
    
    Return:
    ----------
        noise (np.ndarray): noise in the shape of `X`
    '''
    # create noise
    noise_fn = np.random.randn if kind == 'guassian' else np.random.rand
    noise = noise_fn(*X.shape)
    
    # rescale noise to be within certain range
    mms = MinMaxScaler(feature_range=feature_range)
    noise = mms.fit_transform(noise)
    noise *= scale
    return noise


def get_swap_coords(X:np.ndarray) -> Tuple[int, int]:
    '''
    Arguments:
    ----------
        X (np.ndarray): Input data matrix
        
    Returns:
    ----------
        row_loc (int): row of the min of `X`
        col_loc (int): col of the min of `X`
    '''
    row_loc, col_loc = np.where(X == X.min())
    
    row_loc = row_loc[0]
    col_loc = col_loc[0]
    return row_loc, col_loc



def swap(
    X:np.ndarray, 
    i_loc:int, 
    j_loc:int, 
    axis:int=0
) -> np.ndarray:
    '''
    Arguments:
    ----------
        X (np.ndarray): Input data matrix
        
        i_loc (int): The desired row / column location to swap to
        
        j_loc (int): The desired row / column to swap
        
        axis (int): Whether to swap rows (`axis=0`) or columns (`axis=1`)
    
    Returns:
    ----------
        X (np.ndarray): The input matrix after the specified swap
    '''
    x = X.copy()
    if axis == 0:
        x[[i_loc, j_loc], :] = x[[j_loc, i_loc], :]
    elif axis == 1:
        x[:, [i_loc, j_loc]] = x[:, [j_loc, i_loc]]
    else:
        raise ValueError(f'swap only works on 2D matrices')    
    return x



def anchor_min_to_00(
    X:np.ndarray, 
    coords:Tuple[int, int]=None
):
    '''
    Utility function for readability
    
     Arguments:
    ----------
        X (np.ndarray): Input data matrix
        
        coords (Tuple[int, int]): Defaults to `None`. The row and column indicies
            that specifying the minimum element of `X` that needs to be moved to 
            the location `[0, 0]`. If `coords=None`, min will be calculated from `X`.
            If `coords` are provided, will apply this transformation on `X` without
            checking. Note this is used to swap the noise accordingly in our 
            fundamental domain projection function.
               
    Returns:
    ----------
        X (np.ndarray): The input matrix where the min element (or the element located 
            at the given `coords`) are moved to `[0, 0]`.
    
    '''
    x = X.copy()
    if coords is None:
        r, c = get_swap_coords(x)
    else:
        r, c = coords
    x = swap(x, 0, r, axis=0)
    x = swap(x, 0, c, axis=1)
    return x




def sort_first_row_first_col(
    X:np.ndarray, 
    return_order:bool=True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Sort `X` ascending by the first row and first column values.
    
     Arguments:
    ----------
        X (np.ndarray): Input data matrix
        
        return_order (bool): Defaults to `True`. Whether or not to return the sort oders
            for the columns and rows.
               
    Returns:
    ----------
    
        X (np.ndarray): The input matrix where the min element (or the element located 
            at the given `coords`) are moved to `[0, 0]`.
            
        first_row_column_order (np.ndarray): Only returned if `return_order=True`. The
            order in which to sort the columns based on first row.
        
        first_column_row_order (np.ndarray): Only returned if `return_order=True`. The
            order in which to sort the rows based on first column.            
    '''
    row1_col_order = X[0].argsort()
    col1_row_order = X[:, 0].argsort()
    x = X[:, row1_col_order][col1_row_order, :]
    if return_order:
        return x, row1_col_order, col1_row_order
    return x


def sort_rows(
    X:np.ndarray, 
    by:str='sum', 
    axis:int=1, 
    return_order:bool=False
)->np.ndarray:
    '''
    Sort `X`'s rows by the specified numpy function applied on a given axis.
    
     Arguments:
    ----------
        X (np.ndarray): Input data matrix
        
        by (str): Function to apply on `X`. Choices include:
            `'sum'`, `'min'`, `'max'`, `'mean'`.
        
        axis (int): Defaults to `1`. The axis on which to apply the function specified
            by `by` on. Note `X` is still row sorted by the result.
    
        return_order (bool): Defaults to `True`. Whether or not to return the sort oders
            for the columns and rows.
               
    Returns:
    ----------
        X (np.ndarray): The input matrix where the min element (or the element located 
            at the given `coords`) are moved to `[0, 0]`. Only returned if `return_order=False`.
            
        sort_order (np.ndarray): The order in which to sort the rows. 
            Only returned if `return_order=True`.         
    '''
    _valid_by = 'sum min max mean'.split()
    if by not in _valid_by:
        raise ValueError(f'by={by} is unknown. Try one of {_valid_by}')
    
    fn = getattr(np, by)
    sort_order = fn(x2, axis=axis).argsort()
        
    if return_order:
        return sort_order
    
    return X[sort_order]



def fundamental_domain_projection(
    X:np.ndarray, 
    kind:str='guassian', 
    feature_range:Tuple[float, float]=(-0.001, 0.001), 
    scale:float=1
) -> np.ndarray:
    '''
    Moves the smallest element of X to the upper left corner and then
    sorts by first row and first column.
    
    Arguments:
    ----------
        X (np.ndarray): Input data matrix
        
        kind (str): Defaults to `'guassian'`. The kind of noise to generate.
        
        feature_range (Tuple[float, float]): Feature range to limit the noise to.
            Defaults to `(-0.01, 0.01)`.
        
        scale (float): Defaults  to `1`. A post min-max re-scaling via simple 
            multiplication to give further control over the noise.
    
    Return:
    ----------
        projected (np.ndarray): `X` projected.
    '''

    
    # STEP 1: add small perturbation to X to make all entries distinct
    n = create_perturbation(X, kind='guassian', feature_range=feature_range, scale=scale)
    # NOTE: keep n separate because we need to remove it later
    x1 = X + n
    
    # STEP 2: Move smallest entry to upper left corner
    r, c = get_swap_coords(x1)
    
    x2 = anchor_min_to_00(x1, (r, c))
    n2 = anchor_min_to_00(n, (r, c))
    
    # STEP 3: Sort first row and first column 
    x3, co, ro = sort_first_row_first_col(x2, return_order=True)
    n3 = n2[:, co][ro, :]
    
    # STEP 4: undo perturbation
    x = x3 - n3
    return x


def dirichlet_projection(
        X: np.ndarray,
        x0: any=None,
        gen_name:str='neighbourtranspositions',
        seeded_ascent:bool=False,
) -> np.ndarray:
    '''
    Uses gradient ascent to approximate projection onto a fundamental domain around a fixed point x0.

    Arguments:
    ----------
        X (np.ndarray): Input data matrix

        x0 (matrix/str): Fixed point. Default to a random point. 'Daniel' supports lexicographical ordering

        gen_name (str): method used to create generating set. Defaults to 'neighbourtranspositions'

        seeded_ascent (bool): Whether to use gradient ascent with multiple seeds as described in E.2. Defaults to 'false'

    Return:
    ----------
        projected (np.ndarray): `X` projected.
    '''

    X = DirichletDataset(X=[X], matrix_dim=X.shape).X_proj[0]
    return X


if __name__ == '__main__':
    x = np.array([
        [5, 3, 3],
        [4, 0, 0],
        [3, 5, 1]
    ])
    xp = fundamental_domain_projection(x)
    # xp = dirichlet_projection(x)
    print('starting matrix')
    print(x)
    print('Transformed matrix')
    print(xp)