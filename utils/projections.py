import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Union, Tuple
from functools import cmp_to_key

from utils.generating_sets import *
from utils.matrix_auxiliaryfunctions import *

def create_perturbation(
        X: np.ndarray,
        kind: str = 'guassian',
        use_minmax: bool = False,
        feature_range: Tuple[float, float] = (-0.01, 0.01),
        scale: float = 1
) -> np.ndarray:
    '''
    Arguments:
    ----------
        X (np.ndarray): Input data matrix

        kind (str): Defaults to `'guassian'`. The kind of noise to generate.

        use_minmax (bool): Defaults to `False`. Whether or not to use `MinMaxScaler`
            to rescale the noise to be within the range of `feature_range` prior
            to multiplying by `scale`.

        feature_range (Tuple[float, float]): Feature range to limit the noise to.
            Defaults to `(-0.01, 0.01)`.

        scale (float): Defaults  to `1`. A post min-max re-scaling via simple
            multiplication to give further control over the noise.

    Return:
    ----------
        noise (np.ndarray): noise in the shape of `X`
    '''
    # create noise
    noise_fn = np.random.randn if kind == kind else np.random.rand
    noise = noise_fn(*X.shape)

    # rescale noise to be within certain range
    mms = MinMaxScaler(feature_range=feature_range)
    noise = mms.fit_transform(noise)
    noise *= scale
    return noise


def get_swap_coords(X: np.ndarray) -> Tuple[int, int]:
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
        X: np.ndarray,
        i_loc: int,
        j_loc: int,
        axis: int = 0
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
        X: np.ndarray,
        coords: Tuple[int, int] = None
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
        X: np.ndarray,
        return_order: bool = True
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
        X: np.ndarray,
        by: str = 'sum',
        axis: int = 1,
        return_order: bool = False
) -> np.ndarray:
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


def combinatorial_domain_projection(
        X: np.ndarray,
        noise: np.ndarray = None,
        kind: str = 'guassian',
        use_minmax: bool = True,
        feature_range: Tuple[float, float] = (-0.001, 0.001),
        scale: float = 1
) -> np.ndarray:
    '''
    Moves the smallest element of X to the upper left corner and then
    sorts by first row and first column.

    Arguments:
    ----------
        X (np.ndarray): Input data matrix
        noise (np.ndarray): Noise matrix to apply to X. Defaults to `None`.
            When `None` the function `create_perturbation` will be called to
            make a random noise matrix.

        kind (str): Defaults to `'guassian'`. The kind of noise to generate.
        use_minmax (bool): Defaults to `True`. Whether or not to use `MinMaxScaler`
            to rescale the noise to be within the range of `feature_range` prior
            to multiplying by `scale`.

        feature_range (Tuple[float, float]): Feature range to limit the noise to.
            Defaults to `(-0.01, 0.01)`.

        scale (float): Defaults  to `1`. A post min-max re-scaling via simple
            multiplication to give further control over the noise.

    Return:
    ----------
        projected (np.ndarray): `X` projected.
    '''

    # STEP 1: add small perturbation to X to make all entries distinct
    if noise is None:
        n = create_perturbation(
            X,
            kind=kind, use_minmax=use_minmax,
            feature_range=feature_range, scale=scale
        )
    else:
        n = noise
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


def create_fixed_point(
        X: np.ndarray,
        low: int = -8,
        high: int = 6,
        seed: int = 99,
        add_noise: bool = True,
) -> np.ndarray:
    '''
    Arguments:
    ----------
        X (np.ndarray): Input data matrix

        low (int): Lowest possible integer in the fixed matrix. Defaults to `-8`, lowest observed in the dataset.

        high (int): Highest possible integer in the fixed matrix. Defaults to `6`, highest observed in the dataset.

        seed (int): Random seed used to generate fixed matrix. Allows control over fixed matrix generation. Defaults to `99`.

        add_noise (bool): If True, fixed matrix will be a random non-integer matrix else it wil be a random integer matrix.
            Defaults to `True`.

    Return:
    ----------
        x0 (np.ndarray): random fixed point
    '''

    np.random.seed(seed)
    x0 = np.random.randint(low, high, size=X.shape)
    dx0 = np.random.randn(*X.shape)
    if add_noise: x0 += dx0
    print('Fixed point (x0) created: ', x0)
    return x0

def gradient_ascent(
        x: np.ndarray,
        use_ordering: str = 'lexicographical',
        fixed_point: np.ndarray = None,
        generator_set: dict = None,
        generator_type: str = 'neighbourtranspositions',
) -> np.ndarray:
    '''
    Gradient ascent method to approximate Dirichlet projection.

    Arguments:
    ----------
        x (np.ndarray): Input data matrix.

        use_ordering (str): Ordering used to compare two matrices in gradient ascent method for approximating
                Dirichlet projection. Defaults to `lexicographical`.

        fixed_point (np.ndarray): If specified, uses given fixed point to calculate Dirichlet projection.
            Defaults to `None`.

        generator_set (dict): Generator set used in gradient ascent. Defaults to `None`.

        generator_type (str): Generator type used to create generator set for the gradient ascent method for
            Dirichlet projection. Defaults to `neighbourtranspositions`.

    Return:
    ----------
        projected (np.ndarray): `X` projected.
    '''

    x0 = create_fixed_point(x) if use_ordering == 'innerproduct' and fixed_point is None else fixed_point

    if generator_set is None:
        generator_set = {
            'row': generators(generator_type, x.shape, 'row').elements,
            'col': generators(generator_type, x.shape, 'col').elements,
            'trans': generators(generator_type, x.shape, 'trans').elements
        }

    gen_row, gen_col, trans = generator_set['row'], generator_set['col'], generator_set['trans']
    len_row, len_col, len_trans = len(gen_row), len(gen_col), len(trans)

    max_found = False
    while max_found == False:
        i = 0
        while i < len_row + len_col + len_trans:
            if i < len_row:
                y = np.dot(gen_row[i], x)
            if i >= len_row and i < len_row + len_col:
                y = np.dot(x, gen_col[i - len_row])
            if i > len_row + len_col:
                y = trans[i - len_row - len_col](x)
            if matrix_order(x, y, use_ordering=use_ordering, x0=x0) == -1:  # In this case y>x
                x = y
                i = 0
            else:
                i = i + 1
        max_found = True
    return x


def seeded_gradient_ascent(
        x: np.ndarray,
        use_ordering: str = 'lexicographical',
        fixed_point: np.ndarray = None,
        generator_set: dict = None,
        generator_type: str = 'neighbourtranspositions',
) -> np.ndarray:
    '''
    Seeded Gradient ascent method to approximate Dirichlet projection.

    Arguments:
    ----------
        x (np.ndarray): Input data matrix.

        use_ordering (str): Ordering used to compare two matrices in gradient ascent method for approximating
                Dirichlet projection. Defaults to `lexicographical`.

        fixed_point (np.ndarray): If specified, uses given fixed point to calculate Dirichlet projection.
            Defaults to `None`.

        generator_set (dict): Generator set used in gradient ascent. Defaults to `None`.

        generator_type (str): Generator type used to create generator set for the gradient ascent method for
            Dirichlet projection. Defaults to `neighbourtranspositions`.

    Return:
    ----------
        projected (np.ndarray): `X` projected.
    '''
    k, m = x.shape

    x0 = create_fixed_point(x) if use_ordering == 'innerproduct' and fixed_point is None else fixed_point

    if generator_set is None:
        generator_set = {
            'row': generators(generator_type, x.shape, 'row').elements,
            'col': generators(generator_type, x.shape, 'col').elements,
            'trans': generators(generator_type, x.shape, 'trans').elements
        }

    f = lambda y: gradient_ascent(y, use_ordering=use_ordering, fixed_point=x0, generator_set=generator_set)
    seeded_ascents = np.array([[f(np.dot(np.dot(cycle(i, k), x), cycle(j, m))) for i in range(k)] for j in range(m)])

    if use_ordering == 'lexicographical':
        seeds_sorted = sorted(flatten_onelevel(seeded_ascents), key=cmp_to_key(lambda x, y: matrix_order(x, y, x0)))
        return seeds_sorted[-1]
    elif use_ordering == 'innerproduct':
        array_of_norms = np.array(
            [[matrix_innerproduct(seeded_ascents[j][i], x0) for i in range(k)] for j in range(m)])
        max_seed = argmax_nonflat(array_of_norms)
        return seeded_ascents[max_seed]
    else:
        raise ValueError(f'Invalid ordering {use_ordering}. Only supports "lexicographical" and "innerproduct"')


def dirichlet_domain_projection(
        X: np.ndarray,
        use_ordering: str = 'lexicographical',
        fixed_point: np.ndarray = None,
        generator_set: dict = None,
        generator_type: str = 'neighbourtranspositions',
        seeded_ascent: bool = False,
) -> np.ndarray:
    '''
    Moves the smallest element of X to the upper left corner and then
    sorts by first row and first column.

    Arguments:
    ----------
        X (np.ndarray): Input data matrix.

        use_ordering (str): Ordering used to compare two matrices in gradient ascent method for approximating
                Dirichlet projection. Defaults to `lexicographical`.

        fixed_point (np.ndarray): If specified, uses given fixed point to calculate Dirichlet projection.
            Defaults to `None`.

        generator_set (dict): Generator set used in gradient ascent. Defaults to `None`.

        generator_type (str): Generator type used to create generator set for the gradient ascent method for
            Dirichlet projection. Defaults to `neighbourtranspositions`.

        seeded_ascent (bool): Whether to use gradient ascent with multiple starting points. Defaults to `False`.

    Return:
    ----------
        projected (np.ndarray): `X` projected.
    '''

    kwargs = {'use_ordering': use_ordering, 'fixed_point': fixed_point, 'generator_set': generator_set, 'generator_type': generator_type}
    X = seeded_gradient_ascent(X, **kwargs) if seeded_ascent else gradient_ascent(X, **kwargs)
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