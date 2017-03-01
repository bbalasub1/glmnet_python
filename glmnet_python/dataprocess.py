#!/usr/bin/env python

# ----------------------------------------
# glmnet-python
# ----------------------------------------
# a module for data processing
# ----------------------------------------
# author: Han Fang
# contact: hanfang.cshl@gmail.com
# website: hanfang.github.io
# date: 3/1/2017
# ----------------------------------------

import sys
import os
import numpy as np
from scipy import sparse

class dataprocess(object):
    """
    data process module
    """
    def __init__(self):
        """
        """
        pass

    def sparseDf(self, df, matrixType="csc"):
        """
        convert a pandas sparse df to numpy sparse array
        :param df: pandas sparse df
        :param matrixType: csc or csr
        :return: numpy sparse array
        """
        columns = df.columns
        dat, rows = map(list, zip(
            *[(df[col].sp_values - df[col].fill_value, df[col].sp_index.to_int_index().indices) for col in columns]))
        cols = [np.ones_like(a) * i for (i, a) in enumerate(dat)]
        datF, rowsF, colsF = np.concatenate(dat), np.concatenate(rows), np.concatenate(cols)
        arr = sparse.coo_matrix((datF, (rowsF, colsF)), df.shape, dtype=np.float64)
        if matrixType == "csc":
            return arr.tocsc()
        elif matrixType == "csr":
            return arr.tocsc()
        else:
            raise ValueError("Only accept csc or csr")

def main():
    dataprocess.sparseDf()

if __name__ == '__main__':
    main()