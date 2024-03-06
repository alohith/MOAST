#!/usr/bin/env python
import sys, os, subprocess
from datetime import date
from io import StringIO
import pandas as pd, numpy as np
import time, timeit, cython
from numba import jit, prange, njit
import KDEpy as kde
import math
import pyarrow as pa


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
            kde.FFTKDE(bw="silverman", kernel="gaussian").fit(x).evaluate(gridsize)
        )
    except ValueError:
        # print(data[col].describe())
        kde_support, kde_pdf = (
            kde.FFTKDE(bw="silverman", kernel="gaussian")
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
        classesDf: pd.DataFrame = None,
        on=None,
        classesCol=None,
    ) -> None:
        self.dataset = dataset.fillna(0)
        self.nullSet = self.dataset if nullData is None else nullData
        self.nullOption = nullOption
        self.noiseModel = noiseModel
        self.refDist = None
        self.classesDf = classesDf

        self.on = on
        self.classesCol = classesCol

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

    def _getClasses(self, left: pd.DataFrame, on: str, classCol: str):
        mergeDf = pd.merge(
            left=left,
            right=self.classesDf[[on, classCol]],
            left_index=True,
            right_on=on,
            how="left",
        )
        mergeDf.reset_index(inplace=True, drop=True)
        mergeDf.drop([on], axis=1, inplace=True)
        mergeDf = mergeDf.set_index(classCol)
        return mergeDf

    def build(self, distance=True):
        nullModel = self._createNull().fillna(0)

        refDist = _pairwiseCorrProcess(
            exp_df=self.dataset, ref_df=nullModel, distance=distance
        )
        refDist = pd.DataFrame(refDist.tolist(), index=refDist.index).T
        refDist.index = nullModel.index

        ###### TODO: ADD CLASS AGG ######
        if self.classesDf is not None:
            refDist = self._getClasses(
                left=refDist, on=self.on, classCol=self.classesCol
            )
            # print(refDist)
            refDist = refDist.groupby(level=self.classesCol).mean()
        self.refDist = refDist

        ###### TODO: Dictionary {className: KDEsupport, PDF} ######
        kdeDict = {}
        for name, row in self.refDist.iterrows():
            kdeRes = kde.FFTKDE(bw="silverman", kernel="gaussian")
            kdeRes.fit(
                np.nan_to_num(row.to_numpy().flatten()),
                np.linspace(0, 2, len(row)),
            )
            kde_support, kde_pdf = kdeRes.evaluate(len(row))
            kdeDict[name] = (kde_support, kde_pdf)

        ###### TODO: ADD pickle (in MOAST class) ######

        return kdeDict

    def to_pickle(self, outPath, **kwargs):
        from . import MoastException  # prevents circular import

        if self.refDist is not None:
            self.refDist.to_pickle(outPath, **kwargs)
        else:
            raise MoastException(message="This class has not been built yet")

    @property
    def getRefDist(self):
        return self.refDist
