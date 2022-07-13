import numpy as np
from functools import cmp_to_key
from fundamental_domain_projections.dirichlet.generating_sets import *
from fundamental_domain_projections.matrix_permutation_auxiliaryfunctions import *


class DirichletDataset():
    def __init__(self, X, Y, matrix_dim, x0=None, gen_name='neighbourtranspositions', seeded_ascent=False):
        super().__init__()
        self.X = X
        self.Y = Y
        self.matrix_dim = matrix_dim
        self.gen_name = gen_name
        self.seeded_ascent = seeded_ascent
        self.x0 = self._get_x0() if x0 is None else x0
        self.gen_row = generators(gen_name,matrix_dim,'row').elements
        self.gen_col = generators(gen_name,matrix_dim,'col').elements
        self.trans = generators(gen_name, matrix_dim, 'trans').elements
        self.X_proj = self.get_projection(X)

    def _get_x0(self, seed=10):
        np.random.seed(seed)
        x = np.random.randint(-8, 6, size=self.matrix_dim)
        dx = np.random.rand(self.matrix_dim[0], self.matrix_dim[1])
        return x + dx

    def gradient_ascent(self, x, x0):
        len_row = len(self.gen_row)
        len_col = len(self.gen_col)
        len_trans = len(self.trans)

        max_found = False
        while max_found == False:
            i = 0
            while i < len_row + len_col + len_trans:
                if i < len_row:
                    y = np.dot(self.gen_row[i], x)
                if i >= len_row and i < len_row + len_col:
                    y = np.dot(x, self.gen_col[i - len_row])
                if i > len_row + len_col:
                    y = self.trans[i - len_row - len_col](x)
                if matrix_order(x, y, x0) == -1:  # In this case y>x
                    x = y
                    i = 0
                else:
                    i = i + 1
            max_found = True
        return x

    def gradient_ascent_seeded(self, x, x0):
        k, m = self.matrix_dim
        x_seeds = np.array([[np.dot(np.dot(cycle(i, k), x), cycle(j, m)) for i in range(k)] for j in range(m)])
        seeded_ascents = np.array([[self.gradient_ascent(x, x0) for x in row] for row in x_seeds])
        if x0 == 'Daniel':
            seeds_sorted = sorted(flatten_onelevel(seeded_ascents), key=cmp_to_key(lambda x, y: matrix_order(x, y, x0)))
            return seeds_sorted[-1]
        else:
            array_of_norms = np.array(
                [[matrix_innerproduct(seeded_ascents[j][i], x0) for i in range(k)] for j in range(m)])
            max_seed = argmax_nonflat(array_of_norms)
            return seeded_ascents[max_seed]

    def get_projection(self, X):
        X_proj = []
        for x in X:
            x_proj = self.gradient_ascent_seeded(x, self.x0) if self.seeded_ascent else self.gradient_ascent(x, self.x0)
            X_proj.append(x_proj)
        return np.array(X_proj)

if __name__ == "__main__":
    X = np.array([[0, 0, 3, 0, 2, 0, 6, 0, 0], [9, 0, 0, 3, 0, 5, 0, 0, 1], [0, 0, 1, 8, 0, 6, 4, 0, 0],
                  [0, 0, 8, 1, 0, 2, 9, 0, 0], [7, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 6, 7, 0, 8, 2, 0, 0],
                  [0, 0, 2, 6, 0, 9, 5, 0, 0], [8, 0, 0, 2, 0, 3, 0, 0, 9], [0, 0, 5, 0, 1, 0, 3, 0, 0]])
    Y = 10
    dirc_proj = DirichletDataset([X], [Y], X.shape)
    # dirc_proj = DirichletDataset([X], [Y], X.shape, seeded_ascent=True)
    # dirc_proj = DirichletDataset([X], [Y], X.shape, x0="Daniel")
    # dirc_proj = DirichletDataset([X], [Y], X.shape, x0="Daniel", seeded_ascent=True)
    print("Original X: ", dirc_proj.X)
    print("Projection of X: ", dirc_proj.X_proj)