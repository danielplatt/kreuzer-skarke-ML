import numpy as np
import os


def get_data(datasetname='original.npz'):
    '''
    Loads the preprocessed Kreuzer-Skarke dataset

    :param datasetname: 'original.npz' or 'dirichlet_permuted.npz' or 'dirichlet.npz' etc
    :return: An npz file object, use as get_data()['x_proj'] and get_data()['y']
    '''
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'projections/{datasetname}')
    return np.load(filepath)


if __name__ == '__main__':
    print(f'Available indices: {get_data().files}')
    print(f'Shape of x_proj: {get_data()["x_proj"].shape}')
    print(f'Shape of y: {get_data()["y"].shape}')
