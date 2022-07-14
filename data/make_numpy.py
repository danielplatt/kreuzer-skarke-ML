import os, sys, json, gzip, itertools, pickle, numpy as np, pandas as pd
from typing import TypeVar, Union, List, Literal


# NOTE: from jupyter notebook
_file = os.path.abspath('.')
DATA_DIR = os.path.join(_file, 'data')

# NOTE: for this unstructured collection of python files
here = os.path.realpath(__file__)
DATA_DIR = os.path.join(os.path.dirname(here))

RAW_FILE = os.path.join(DATA_DIR, 'raw', 'v26.gz')
UNZIPPED_FILE = os.path.join(DATA_DIR, 'raw', 'v26')


kreuzer_skarke_description_header_test_cases = [
    '4 26  M:51 26 N:11 10 H:6,46 [-80]\n',
    '4 26  M:48 26 N:12 11 H:7,43 [-72]\n',
]

KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS = '''
rows cols lattice_points_convex_hull vertices_convex_hull
lattice_points_polar_dual_polytope vertices_polar_dual_polytope
h_1,1 h_1,2 topological_euler_characteristic id
'''.split()

def unzip(file_in:str, file_out:str=None) -> None:
    '''
    Unzip the provided file. Equivalent to bash command `$ gunzip file_in`
    
    Notes:
    ----------
        - assumes `file_in` is a gunzip file, e.g. `file.gz`
    
    Arguments:
    ----------
        file_in (str): Path to input file.
        
        file_out (str): Name of file to save uncompressed version as. Defaults to `None`.
            If `file_out=None` will be `file_in` without the extension (e.g. `file.gz` --> `file`)
        
    Returns:
    ----------
        None
        
    '''
    if file_out is None:
        file_out = os.path.splitext(file_in)[0]
    with gzip.open(file_in, 'rb') as fi, open(file_out, 'w') as fo:
        for line in fi:
            fo.write(line.decode())


def list_to_idx_dict(arr:list)->dict:    
    '''
    Helper function that creates an enumerated dictionary
    where each elemenet of the list is the value of the corresponding index
    
    Arguments:
    ----------
        arr (list): A list of any type
        
    Returns:
    ----------
        res (dict): A dictionary of form `{i: element}`, where `i` is the index of `element`
            from the passed `arr`
    '''
    return dict(zip(range(len(arr)), arr))

def assert_kreuzer_skarke_header(header:str) -> None:
    assert 'M:' in header, 'Convex Hull specification missing'
    assert 'N:' in header, 'Polar Dual Polytope specification missing'
    assert 'H:' in header, 'Hodge Numbers specification missing'
    assert '[' in header and ']' in header, 'Topological Euler characteristic of X specification missing'
    
def split_kreuzer_skarke_header(header:str) -> list[int]:
    '''
    Arguments:
    ----------
        header (str): A string containing a Kreuzer-Skarke description header. For more details
            see http://www2.macaulay2.com/Macaulay2/doc/Macaulay2-1.18/share/doc/Macaulay2/ReflexivePolytopesDB/html/___Kreuzer-__Skarke_spdescription_spheaders.html
                    
    Returns:
    ----------
        parts (list[int]): A list of ints corresponding to the values of each element in the header.
    '''
    parts = header.rstrip().split(' ')
    
    # Remove empty strings due to tabs or other export errors
    parts = list(filter(lambda e: e, parts))
    
    # Remove M:, N:, H:
    parts = list(map(lambda e: e.split(':')[-1], parts))
    
    # Remove brackets around topological Euler characteristic of X (where X is the Calabi-Yau variety)
    parts = list(map(lambda e: e.lstrip('[').rstrip(']'), parts))
    
    # Separate the two Hodge numbers
    parts = list(map(lambda e: e.split(','), parts))
    
    # Reflatten into a list
    parts = list(itertools.chain(*parts))
    
    # Type cast to int
    try:
        parts = list(map(int, parts))
    except ValueError:
        raise ValueError('Not all elements are ints')
    # By definition (see docstring) header has no more than 10 elements and default labels has 10 elements
    assert len(parts) in [9, 10], 'Kreuzer-Skarke description header has only 9 or 10 values'
    return parts

def is_kreuzer_skarke_header(header:str) -> bool:
    '''
    Helper function to test if a line in a file matches Kreuzer-Skarke header
    specifications
    
    Arguments:
    ----------
        header (str): A presumed Kreuzer-Skarke header
        
    Returns:
    ----------
        res (bool): Whether or not the str is in fact a header
    '''
    try:
        assert_kreuzer_skarke_header(header)        
        parts = split_kreuzer_skarke_header(header)
    except AssertionError:
        return False
    return True

def is_valid_nonheader_line(line:str) -> bool:
    '''
    Helper function to test if a line in a file could be the line of a 
    matrix i.e. only consists of floats
    
    Arguments:
    ----------
        header (str): A presumed (part of a) matrix in string form
        
    Returns:
    ----------
        res (bool): Whether or not the str contains only float elements
    '''
    try:
        to_float = list(map(float, line.split()))
        return True
    except ValueError:
        return False
    return True

def parse_kreuzer_skarke_description_header(
    header:str, 
    labels:list[str]=KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS
) -> dict:
    '''
    Arguments:
    ----------
        header (str): A string containing a Kreuzer-Skarke description header. For more details
            see http://www2.macaulay2.com/Macaulay2/doc/Macaulay2-1.18/share/doc/Macaulay2/ReflexivePolytopesDB/html/___Kreuzer-__Skarke_spdescription_spheaders.html
        
        labels (list[str]): A list of strings corresponding to the names of each element in the `header`.
            Defaults to `KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS`
            
    Returns:
    ----------
        res (dict): A dictionary containing the labeled values from the `header` i.e. `{label: value}`.
    '''
    # {i: label} pairs
    label_dict = list_to_idx_dict(labels)  
    
    # [el1, el2, ...] 
    parts = split_kreuzer_skarke_header(header)
    
    # {i: el} pairs
    parts_dict = list_to_idx_dict(parts)
    
    # {label_i: el_i} pairs
    labeled_parts = {label_dict[i]: v for i, v in parts_dict.items()}
    return labeled_parts
 
    
def parse_matrix_string(
    header:dict, 
    matrix_string:str, 
    labels:list[str]=KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS
) -> np.ndarray:
    '''
    Arguments:
    ----------
        header (dict): Results from the function `parse_kreuzer_skarke_description_header` i.e. a 
            dictionary of `{label: value}` pairs where `label` is an element from `labels` and
            value is an integer. 
            
        matrix_string (str): A string consisting of only floats. Should have n_rows x n_cols values
            as specified by the passed `header` dict.
            
        labels (list[str]): A list of strings corresponding to the names of each element in the `header`.
            Defaults to `KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS`
            
    Returns:
    ----------
        matrix (np.ndarray): A numpy array with n_rows x n_cols values as specified in the passed `header`
    '''
    n_rows_label, n_cols_label, *_ = labels
    matrix = np.array(list(map(float, matrix_string.split())))
    matrix = np.reshape(matrix, (header[n_rows_label], header[n_cols_label])).astype(float)
    return matrix
    
def read_kreuzer_skarke_file(
    file:str, 
    labels:list[str]=KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS
) -> (list[dict], list[np.ndarray]):
    '''
    Notes:
    ----------
        - We assume ALL matrices in the file share the same shape, e.g. `(4, 26)`
    
    
    Arguments:
    ----------
        file (str): Path to input file.
            
        labels (list[str]): A list of strings corresponding to the names of each element in the `header`.
            Defaults to `KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS`
            
    Returns:
    ----------
        headers (list[dict]): A list of the headers from the provided `file`.
        matrices (list[np.ndarray]): A list of the matrices from the provided `file`.
    '''
    with open(file, 'r') as f:
        # Tracker variables
        header = None
        matrix_string = '' 
        
        # Storage variables
        headers = []
        matrices = []
        
        for line in f:
            
            is_header_line = is_kreuzer_skarke_header(line)
            
            if is_header_line:
                # NOTE: handle condition where we are reading a header other than the first
                #       i.e. there should be a matrix_string and header already defined
                if matrix_string != '':                                        
                    matrix = parse_matrix_string(header, matrix_string, labels)
                    matrices.append(matrix)
                
                # NOTE: no matter what append new header to storage variable
                header = parse_kreuzer_skarke_description_header(line)
                headers.append(header)
                  
                # NOTE: we have read either the first or a new header and we have already 
                #       handled non-empty string matrix strings 
                matrix_string = ''  

                
            # NOTE: otherwise check if line is all float and aggregate to convert to matrix
            else:
                is_matrix_line = is_valid_nonheader_line(line)
                if not is_matrix_line:
                    continue
                matrix_string += line
        
        # NOTE: since matricies come after headers, we should have one last matrix to append
        if len(headers) > len(matrices):
            assert len(headers) == len(matrices) + 1
            matrix = parse_matrix_string(header, matrix_string, labels)
            matrices.append(matrix)
           
    return headers, matrices
 
    
def extract_from_headers(
    headers:list[dict], 
    key:str, 
    labels:list[str]=KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS
) -> list[int]:
    '''
    Utility function to extract a labeled value from all headers in a list.
    
    Arguments:
    ----------
        headers (list[dict]): A list of the kreuzer-skarke headers in dictionary format.
            
        key (str): The label to extract.
        
        labels (list[str]): A list of strings corresponding to the names of each element in the `header`.
            Defaults to `KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS`
            
    Returns:
    ----------
        values (list[int]): A list of the values from the provided `headers`.
    '''
    assert key in labels
    return list(map(lambda e: e[key], headers))

def preprocessing_pipeline(
        file: str = UNZIPPED_FILE,
        extraction_key: str = 'h_1,1',
        labels: list[str] = KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS,
        save: bool = False,
        load: bool = True,
        apply_random_permutation: bool = False
) -> (np.ndarray, np.ndarray):
    '''
    Utility function to extract a labeled value from all headers in a list.
    
    Notes:
    ----------
        - We assume ALL matrices in the file share the same shape, e.g. `(4, 26)`
    
    
    
    Arguments:
    ----------
        file (str): The input file to read. If compressed (i.e. `gunzip`-ed) will attempt to unzip it.
            
        extraction_key (str): The label to extract. If `None` will return the headers.
        
        labels (list[str]): A list of strings corresponding to the names of each element in the `header`.
            Defaults to `KREUZER_SKARKE_DESCRIPTION_HEADER_LABELS`.
            
        save (bool): Whether or not to save final `matricies` and `values`. Defaults to `False`.
        
        load (bool): Whether or not ot try and load final results. Defaults to `True`.

        apply_random_permutation (bool): Whether to apply a random row and column permutation to each input matrix.

    Returns:
    ----------
        matrices (np.ndarray): A list of the matricies from the provided `file`.
        
        values (np.ndarray): A list of the headers from the provided `file` or a list of just the value
            specified by `extraction_key`.         
    '''
    suffix = ''
    if apply_random_permutation:
        suffix = 'permuted'

    if load:
        file = os.path.splitext(file)[0]
        if os.path.isfile(f'{file}_X{suffix}.npy') and os.path.isfile(f'{file}_y.npy'):
            with open(f'{file}_X{suffix}.npy', 'rb') as fx, open(f'{file}_y.npy', 'rb') as fy:
                X = np.load(fx, allow_pickle=True)
                y = np.load(fy, allow_pickle=True)
                return X, y
    
    if '.gz' in file:
        unzip(file, None)
        file = os.path.splitext(file)[0]
            
    headers, matrices = read_kreuzer_skarke_file(file)
    
    
    # NOTE: the default input file `v26.gz` has a several bad lines where the matrix is transposed.
    #       thus to get a standardized numpy tensor we need to transpose it back,
    n_rows_label, n_cols_label, *_ = labels
    zipped = zip(headers, matrices)
    headers = []
    matrices = []
    for header, matrix in zipped:
        r, c = matrix.shape
        
        if r > c:
            matrix = matrix.T
            header[n_rows_label] = c
            header[n_cols_label] = r

        if apply_random_permutation:
            rot_mat = np.random.permutation(matrix)
            double_rot_mat = np.transpose(np.random.permutation(np.transpose(matrix)))
            matrix = double_rot_mat
        headers.append(header)
        matrices.append(matrix)
        
    if extraction_key is not None:
        values = extract_from_headers(headers, extraction_key, labels)
    else:
        values = headers
    
    
    X = np.array(matrices)
    y = np.array(values)
    
    if save:
        file = os.path.splitext(file)[0]
        with open(f'{file}_X{suffix}.npy', 'wb') as fx, open(f'{file}_y.npy', 'wb') as fy:
            np.save(fx, X)
            np.save(fy, y)
    
    return X, y 


if __name__ == '__main__':
    unzip(RAW_FILE, UNZIPPED_FILE)

    X, y = preprocessing_pipeline(UNZIPPED_FILE, save=True, apply_random_permutation=False)
    Xpermuted, _ = preprocessing_pipeline(UNZIPPED_FILE, save=True, apply_random_permutation=True)

    print(X[1])
    print(Xpermuted[1])
