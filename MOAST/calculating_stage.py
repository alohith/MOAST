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

# from .utils import pairwiseCorrProcess
import json
from auc_roc_class_analysis.auc_roc_toolkit import calcSimMat


class Run:
    def __init__(
        self,
        exp_set: pd.DataFrame,
        ref_set: pd.DataFrame,
        kde_dict: dict,
        annots: pd.DataFrame,
        on: str,
        classes_col: str,
    ) -> None:
        self.ref_set = ref_set
        self.exp_set = exp_set
        self.kde_dict = kde_dict
        self.annots = annots
        self.merge_col = on
        self.classes_col = classes_col

        self.ref_set = self._merge_annots(df=self.ref_set)
        self.exp_set = self._merge_annots(df=self.exp_set)

    def _merge_annots(self, df: pd.DataFrame):
        from . import MoastException

        if self.annots is None:
            raise MoastException("annots is empty")

        merge_df = pd.merge(
            left=df,
            right=self.annots[[self.merge_col, self.classes_col]],
            left_index=True,
            right_on=self.merge_col,
            how="left",
        )
        merge_df.set_index([self.merge_col, self.classes_col], inplace=True)

        return merge_df

    def run(self, distance: bool = False):
        sim_matrix = calcSimMat(
            exp_df=self.exp_set, ref_df=self.ref_set, distance=distance
        )

        ref_classes = {
            c: {"comps": [], "avg_dist/sim score": []}
            for c in self.exp_set.index.get_level_values(self.classes_col).unique()
        }
        for name, df in self.exp_set.groupby(level=self.classes_col):
            ref_classes[name]["comps"] = list(df.index.get_level_values(0).values)

        # j_str = json.dumps(ref_classes, indent=4)
        # print(j_str)

        for k, v in ref_classes.items():
            if len(v["comps"]) > 0:
                for comp in v["comps"]:
                    c_comps_mean2Class = np.nanmean(
                        sim_matrix.loc[comp, v["comps"]].values
                    )
                    print(c_comps_mean2Class)
