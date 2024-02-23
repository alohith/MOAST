#!/usr/bin/env python
import sys, os, subprocess
from datetime import date
from io import StringIO
import pandas as pd, numpy as np
import time, timeit, cython
from numba import jit, prange, njit


@jit(nopython=True, parallel=True)
def _corrDist(compSig, refArray):
    def cor(u, v, centered=True):  # ripped from scipy.spatial.distances
        if centered:
            umu = np.average(u)
            vmu = np.average(v)
            u = u - umu
            v = v - vmu
        uv = np.average(u * v)
        uu = np.average(np.square(u))
        vv = np.average(np.square(v))
        dist = 1 - uv / np.sqrt(uu * vv)
        return np.abs(dist)

    num_ref_sigs = refArray.shape[0]
    return_dists = np.empty(num_ref_sigs)

    for i in prange(num_ref_sigs):
        x = refArray[i]
        return_dists[i] = cor(compSig, x)

    return return_dists


@jit(nopython=True, parallel=True)
def _pearsonr(compSig, refArray):
    num_ref_sigs = refArray.shape[0]
    return_corrs = np.empty(num_ref_sigs)

    for i in prange(num_ref_sigs):
        x = refArray[i, :]
        return_corrs[i] = np.corrcoef(x, compSig)[0, 1]
    return return_corrs


def _pairwiseCorrProcess(exp_df, ref_df, distance=True):
    if distance:
        data = exp_df.apply(
            lambda compSig: _corrDist(
                compSig=compSig.to_numpy(), refArray=ref_df.to_numpy()
            ),
            axis=1,
        )
    else:
        data = exp_df.apply(
            lambda compSig: _pearsonr(
                compSig=compSig.to_numpy(), refArray=ref_df.to_numpy()
            ),
            axis=1,
        )
    return data


class Build:
    def __init__(self, dataset: pd.DataFrame, nullData: pd.DataFrame = None) -> None:
        self.dataset = dataset.fillna(0)
        self.nullSet = self.dataset if nullData is None else nullData

    def _createNull(self):
        return self.nullSet.copy().sample(n=15000, replace=True)

    def build(self, distance=True):
        nullModel = self._createNull().fillna(0)
        refDist = _pairwiseCorrProcess(
            exp_df=self.dataset, ref_df=nullModel, distance=distance
        )
        refDist = pd.DataFrame({refDist.index[i]: d for i, d in enumerate(refDist)})
        return refDist


########################### TESTING BLOCK ###########################
# def main():
#     testdf = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/ImmunoCP/designerHD_concats/SP20312_DMSO+LPS-DMSO_noCts_histdiffpyROWSORT.csv"
#     testData = pd.read_csv(testdf, index_col=0)

#     b = Build(dataset=testData)
#     refDist = b.build()
#     print(refDist)


# if __name__ == "__main__":
#     main()

## OUTPUT:
#        BMS-345541_0.016uM  BMS-345541_0.016uM_+_Gardiquimod_0.016uM  ...  VM01.3_+_QNZ-10_0.4uM       NaN
# 0                0.286582                                  0.235619  ...               0.073456  0.212760
# 1                0.242252                                  0.410468  ...               0.234001  0.286011
# 2                0.156114                                  0.304933  ...               0.183598  0.350290
# 3                0.283838                                  0.108880  ...               0.090321  0.178747
# 4                0.264897                                  0.237458  ...               0.132649  0.242708
# ...                   ...                                       ...  ...                    ...       ...
# 14995            0.156114                                  0.304933  ...               0.183598  0.350290
# 14996            0.379619                                  0.177450  ...               0.084122  0.199537
# 14997            0.709235                                  0.959712  ...               0.687754  0.848354
# 14998            0.670603                                  0.516769  ...               0.465857  0.513841
# 14999            0.473693                                  0.467027  ...               0.414266  0.603866

# [15000 rows x 384 columns]
