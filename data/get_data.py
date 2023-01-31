import numpy as np
import os
import pandas
from tempfile import TemporaryFile


def get_data(datasetname='original.npz', one_hot=False):
    '''
    Loads the preprocessed Kreuzer-Skarke dataset

    :param datasetname: 'original.npz' or 'dirichlet_permuted.npz' or 'dirichlet.npz' etc
    :return: An npz file object, use as get_data()['x_proj'] and get_data()['y']
    '''
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'projections/{datasetname}')
    data = np.load(filepath)
    if not one_hot:
        return data

    outfile = TemporaryFile()
    np.savez(outfile, x=data['x'], y=pandas.get_dummies(data['y']).to_numpy(), x_proj=data['x_proj'])
    outfile.seek(0)  # Only needed here to simulate closing & reopening file
    npzfile = np.load(outfile)

    return npzfile


if __name__ == '__main__':
    print('Try loading not one-hot encoded data')
    data = get_data()
    print(f'Available indices: {data.files}')
    print(f'Shape of x_proj: {data["x_proj"].shape}')
    print(f'Shape of y: {data["y"].shape}')

    print('Try loading one-hot encoded data')
    data = get_data(one_hot=True)
    print(f'Available indices: {data.files}')
    print(f'Shape of x_proj: {data["x_proj"].shape}')
    print(f'Shape of y: {data["y"].shape}')
