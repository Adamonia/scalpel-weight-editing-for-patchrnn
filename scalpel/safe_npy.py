"""Restricted loader for object-array .npy files (no arbitrary code on unpickle)."""
import pickle
import numpy as np
from numpy.lib import format as npformat

_ALLOWED = {
    ('numpy.core.multiarray', '_reconstruct'),
    ('numpy._core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'),
    ('numpy._core.multiarray', 'scalar'),
    ('numpy', 'ndarray'),
    ('numpy', 'dtype'),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            module = module.replace('numpy.core', 'numpy._core') if hasattr(np, '_core') else module
            return super().find_class(module, name)
        raise pickle.UnpicklingError('blocked global: %s.%s' % (module, name))


def load_object_npy(path):
    """Load a .npy file. Object arrays are read with a restricted unpickler."""
    with open(path, 'rb') as fp:
        version = npformat.read_magic(fp)
        if version == (1, 0):
            shape, fortran, dtype = npformat.read_array_header_1_0(fp)
        else:
            shape, fortran, dtype = npformat.read_array_header_2_0(fp)
        if dtype.hasobject:
            return _RestrictedUnpickler(fp, encoding='latin1').load()
    return np.load(path, allow_pickle=False)
