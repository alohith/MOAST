#!/usr/bin/env python
import pandas as pd, numpy as np
import sys, os
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


def pairwiseCorrProcess(exp_df, ref_df, distance=True):
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
