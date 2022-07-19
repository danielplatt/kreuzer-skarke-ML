import logging
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset

from src.utils.make_numpy import preprocessing_pipeline
from src.utils.generating_sets import *
from src.utils.projections import *

from config.constants import *

class KreuzerSkarkeDataset(Dataset):
    def __init__(
            self,
            input_file: str = UNZIPPED_INPUT_FILE,
            extraction_key: str = 'h_1,1',
            num_classes: int = None,
            use_one_hot: bool = False,
            apply_random_permutation: bool = False,
            num_samples = -1,
            use_cuda: bool = False,
            logger: logging.Logger = None,

            # --- Fundamental Domain Projection specific arguments ---
            projection: str = None,
            save_projections: bool = False,
            load_projections: bool = False,
            projections_file: str = None,

            # Combinatorial
            use_fixed_perturbation: bool = False,
            perturbation: np.ndarray = None,

            # Dirichlet
            use_ordering: str = 'lexicographical',
            fixed_point: np.ndarray = None,
            generator_type: str = 'neighbourtranspositions',
            seeded_ascent: bool = False,
    ):
        '''
        Arguments:
        ----------
            input_file (str): The input file to read. If compressed (i.e. `gunzip`-ed) will attempt to unzip it.

            extraction_key (str): The label to extract. If `None` will return the headers.

            num_classes (int): Number of classes in `y`. Defaults to `None`. If `None` will be set to
                `max(y)+1`.

            use_one_hot (bool): whether or not to return `y` as a one-hot encoded vector or as its class label

            apply_random_permutation (bool): Whether or not to randomly permute each matrix. Defaults to `False`.

            num_samples (int): Number of matrices to include in the dataset. Useful if only a subset of data is required.
                Defaults to `-1`, i.e. the whole dataset.

            use_cuda (str): Defaults to `True`. Whether or not to put tensors on cuda.

            logger (logging.Logger): Optional logger.


            Projection specific arguments:
            ------------------------------

            projection (str): Funadamental Domain Projection to apply to `X`. Defaults to `None`. If `None`, will
                apply no projection. Currently supports `combinatorial` and `dirichlet` projections.

            save_projections (bool): Whether or not to save projections computed for the input matrix since calculating
                projections is compute intensive. Defaults to `False`.

            load_projections (bool): Whether or not to load pre-calculated projections for the input matrix.
                Defaults to `False`.

            projections_file (str): File used to save or load projections. Defaults to `None`.

            use_fixed_perturbation (bool): Whether or not to use a fixed perturbation for
                when `projection=='combinatorial'. Defaults to `False`.

            perturbation (np.ndarray): Defaults to `None`. When provided will be the fixed
                perturbation used if `use_fixed_perturbation=True`.

            use_ordering (str): Ordering used to compare two matrices in gradient ascent method for aproximating
                Dirichlet projection. Defaults to `lexicographical`.

            fixed_point (np.ndarray): If specified, uses given fixed point to calculate Dirichlet projection.
                Defaults to `None`.

            generator_type (str): Generator type used to create generator set for the gradient ascent method for
                Dirichlet projection. Defaults to `neighbourtranspositions`.

            seeded_ascent (bool): Whether to use gradient ascent with multiple starting points. Defaults to `False`.
        '''

        self.input_file = input_file
        self.extraction_key = extraction_key
        self.num_classes = num_classes
        self.use_one_hot = use_one_hot
        self.apply_random_permutation = apply_random_permutation
        self.num_samples = num_samples
        self.use_cuda = use_cuda
        self.logger = logger

        self.projection = projection
        self.save_projections = save_projections
        self.load_projections = load_projections
        self.projections_file = projections_file

        self.use_fixed_perturbation = use_fixed_perturbation
        self.perturbation = perturbation

        self.use_ordering = use_ordering
        self.fixed_point = fixed_point
        self.generator_type = generator_type
        self.seeded_ascent = seeded_ascent

        if load_projections:
            self._load_projections()
            self.X, self.Y, self.X_proj = self.X[:num_samples], self.Y[:num_samples], self.X_proj[:num_samples]
        else:
            self.X, self.Y = preprocessing_pipeline(input_file,
                                                    extraction_key=extraction_key,
                                                    apply_random_permutation=apply_random_permutation)
            self.X, self.Y = self.X[:num_samples], self.Y[:num_samples]
            self.X_proj = self.X

            self.num_classes = np.max(self.Y) + 1 if num_classes is None else num_classes
            self.matrix_dim = self.X[0].shape

            self.perturbation = create_perturbation(self.X[0]) if perturbation is None else perturbation
            self.fixed_point = create_fixed_point(self.X[0]) if use_ordering=='innerproduct' and fixed_point is None else fixed_point

            if projection is not None:
                assert projection in ['combinatorial', 'dirichlet']
                self.X_proj = self._combinatorial_projection() if projection=='combinatorial' else self._dirichlet_projection()

            if save_projections:
                self._save_projections()


    def _log_msg(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _save_projections(self):
        assert self.projections_file is not None
        rawpath = PROJECTION_DIR.joinpath(self.projections_file)
        print('Saving projection to % s' % rawpath)
        np.savez_compressed(rawpath, x=self.X, y=self.Y, x_proj=self.X_proj)

    def _load_projections(self):
        assert self.projections_file is not None
        rawpath = PROJECTION_DIR.joinpath(self.projections_file+'.npz')
        print('Loading projection from %s ...' % rawpath)
        loaded = np.load(rawpath)
        print('...finished loading')
        self.X, self.Y, self.X_proj = loaded['x'], loaded['y'], loaded['x_proj']

    def _combinatorial_projection(self):
        noise = self.perturbation if self.use_fixed_perturbation else None
        f = lambda x: combinatorial_domain_projection(x, noise=noise)
        print('Calculating combinatorial projections...')
        X_proj = np.array(list(map(f, tqdm(self.X))))
        print('...finished.')
        return X_proj

    def _dirichlet_projection(self):
        gen_set = {
            'row': generators(self.generator_type, self.matrix_dim, 'row').elements,
            'col': generators(self.generator_type, self.matrix_dim, 'col').elements,
            'trans': generators(self.generator_type, self.matrix_dim, 'trans').elements
        }
        f = lambda x: dirichlet_domain_projection(x, use_ordering=self.use_ordering, fixed_point=self.fixed_point,
                                                  generator_set=gen_set, seeded_ascent=self.seeded_ascent)
        print('Calculating dirichlet projections...')
        X_proj = np.array(list(map(f, tqdm(self.X))))
        print('...finished.')
        return X_proj

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X_proj[idx])
        y = torch.as_tensor(self.Y[idx])
        if self.use_one_hot:
            y = nn.functional.one_hot(y, num_classes=self.num_classes)
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        return x, y


if __name__ == "__main__":
    # (a) Original
    # dataset = KreuzerSkarkeDataset(num_samples=10)

    # (b) Combinatorial projection applied to original
    # dataset = KreuzerSkarkeDataset(projection='combinatorial', num_samples=10)

    # (c) Dirichlet projection (with x0=‘Daniel’) applied to original
    # dataset = KreuzerSkarkeDataset(projection='dirichlet', num_samples=10)

    # (d) randomly permuted input
    # dataset = KreuzerSkarkeDataset(apply_random_permutation=True, num_samples=10)

    # (e) Combinatorial projection applied to randomly permuted input
    # dataset = KreuzerSkarkeDataset(projection='combinatorial', apply_random_permutation=True, num_samples=10)

    # (f) Dirichlet projection (with x0=‘Daniel’) applied to randomly permuted input
    # dataset = KreuzerSkarkeDataset(projection='dirichlet', apply_random_permutation=True, num_samples=10)

    # --- Other useful functionalities ---

    # Save projections in numpy compressed format (.npz)
    dataset = KreuzerSkarkeDataset(projection='dirichlet', num_samples=10, save_projections=True, projections_file='test_dirichlet_10')

    # Load projections from numpy compressed format (.npz)
    # dataset = KreuzerSkarkeDataset(load_projections=True, projections_file='test_dirichlet_10')

    print((dataset.X[0]==dataset.X_proj[0]).all())
    # print(dataset.X_proj[0])