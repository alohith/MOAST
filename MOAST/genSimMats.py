#!/usr/bin/env python
import pandas as pd, numpy as np
import sys, os
from numba import jit, prange, njit
from .utils import pairwiseCorrProcess


class GenSimMats:
    """
    Generates Similarity Matrix for given datasets
    """

    def __init__(
        self, expData: pd.DataFrame, refData: pd.DataFrame, distance: bool = True
    ) -> None:
        self.expData = expData
        self.refData = refData
        self.distance = distance

    @property
    def getReportDf(self):
        """Returns the Reporting DF"""
        data = pairwiseCorrProcess(self.expData, self.refData, distance=self.distance)
        return pd.DataFrame(data, index=self.refData.index, columns=self.expData.index)
