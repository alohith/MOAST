#!/usr/bin/env python
import sys, os, subprocess
from datetime import date
from io import StringIO
import pandas as pd, numpy as np
import time, timeit, cython
from numba import jit, prange, njit
import KDEpy as kdeOpt
import math


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


def drawfromfeatureKDE(
    featureData,
    numDraws: int = 15000,
    noiseModel: str = "poisson",
    gridsize: int = 1000,
    logTransform: bool = False,
    col: str = None,
):
    x = np.asarray(featureData)
    if logTransform:
        x = np.log(x)
    try:
        kde_support, kde_pdf = (
            kdeOpt.FFTKDE(bw="silverman", kernel="gaussian").fit(x).evaluate(gridsize)
        )
    except ValueError:
        # print(data[col].describe())
        kde_support, kde_pdf = (
            kdeOpt.FFTKDE(bw="silverman", kernel="gaussian")
            .fit(np.nan_to_num(x))
            .evaluate(gridsize)
        )
    # kde = sm.nonparametric.kde.KDEUnivariate(x)
    # kde.fit(gridsize=gridsize)
    if logTransform:
        kdeChoices = math.e**kde_support
    else:
        kdeChoices = kde_support
    # here the needs to be a support mutator in order to add uniform noise
    scaleWeight = np.var(x)  # or scaleWeight = std(x)  ???
    noise = (1 / gridsize) * scaleWeight
    if noiseModel == "poisson":
        try:
            noise_addition = np.random.poisson(
                lam=scaleWeight ** (1 / gridsize), size=kdeChoices.shape
            )
        except ValueError:
            noise_addition = np.random.poisson(lam=15, size=kdeChoices.shape)
    elif noiseModel == "normal":
        noise_addition = np.random.normal(np.mean(x), np.std(x), kdeChoices.shape)
    elif noiseModel == "normal+":
        noise_addition = np.random.normal(np.mean(x), np.std(x), kdeChoices.shape) + (
            (1 / gridsize) * scaleWeight
        )
    elif noiseModel == "normal_scaledSigma":
        noise_addition = np.random.normal(
            np.mean(x), scaleWeight ** (1 / gridsize), kdeChoices.shape
        )
    elif noiseModel == "scaled":
        noise_addition = np.random.normal(0, 1, kdeChoices.shape) * noise
    elif noiseModel == "unitNormal":
        noise_addition = np.random.normal(0, 1, kdeChoices.shape)
    elif noiseModel == "uniform":
        noise_addition = (
            np.random.uniform(0, 1 / gridsize, kdeChoices.shape) * scaleWeight
        )
    elif noiseModel == None:
        noise_addition = 0.5
    else:
        noise_addition = 0
    kde_pdf += noise_addition
    # return kdeChoices,kde_pdf
    randomDrawsFromKDE = np.empty(numDraws)
    for i in np.arange(0, numDraws):
        try:
            randomDrawsFromKDE[i] = np.random.choice(
                kdeChoices, p=kde_pdf / kde_pdf.sum()
            )
        except ValueError:
            print(col, pd.Series(kde_pdf).describe())
            return np.zeros(numDraws) + noise_addition
    return randomDrawsFromKDE


class Build:
    def __init__(
        self,
        dataset: pd.DataFrame,
        nullData: pd.DataFrame = None,
        nullOption: str = "15KSample",
        noiseModel: str = None,
    ) -> None:
        self.dataset = dataset.fillna(0)
        self.nullSet = self.dataset if nullData is None else nullData
        self.nullOption = nullOption
        self.noiseModel = noiseModel
        self.refDist = None

    def _createNull(self):
        if self.nullOption == "15KSample":
            return self.nullSet.copy().sample(n=15000, replace=True)
        else:

            featCols = self.nullSet.columns
            newRandCPfingerprints = pd.DataFrame(
                index=pd.RangeIndex(15000), columns=featCols
            )
            for col in featCols:
                if self.nullOption == "unitNormal":
                    newRandCPfingerprints[col] = np.random.normal(0, 1, 15000)
                elif self.nullOption == "featureNormal":
                    newRandCPfingerprints[col] = np.random.normal(
                        np.mean(self.nullSet[col]), np.std(self.nullSet[col]), 15000
                    )
                elif self.nullOption == "flattenKDE":
                    # kdeDraws[col] = drawfromfeatureKDE(data[col],100000,col=col)
                    newRandCPfingerprints[col] = drawfromfeatureKDE(
                        self.nullSet[col], 15000, col=col, noiseModel=self.noiseModel
                    )  # noiseModel='normal_scaledSigma')

            return newRandCPfingerprints

    def _getClasses(self):
        pass

    def build(self, distance=True):
        nullModel = self._createNull().fillna(0)
        print(nullModel.index)

        refDist = _pairwiseCorrProcess(
            exp_df=self.dataset, ref_df=nullModel, distance=distance
        )
        refDist = pd.DataFrame(refDist.tolist(), index=refDist.index).T
        refDist.index = nullModel.index
        self.refDist = refDist

        ###### TODO: ADD CLASS AGG ######
        ###### TODO: Dictionary className: KDEsupport, PDF ######

        ###### TODO: ADD pickle (in MOAST class) ######
        return refDist


########################### TESTING BLOCK ###########################
# def main():
#     # testdf = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/ImmunoCP/designerHD_concats/SP20312_DMSO+LPS-DMSO_noCts_histdiffpyROWSORT.csv"
#     testdf = "/Users/dterciano/Desktop/LokeyLabFiles/ImmunoCP/designerHD_concats/LPS-DMSO_longConcat_hd.csv"
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