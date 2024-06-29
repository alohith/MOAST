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

        ########### TEMP CODE ###########
        class_counts = self.exp_set.index.get_level_values(
            self.classes_col
        ).value_counts()
        self.exp_set = self.exp_set[
            self.exp_set.index.get_level_values(self.classes_col).isin(
                class_counts[class_counts > 3].index
            )
        ]

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

    def run(self, distance: bool = True) -> pd.DataFrame:
        sim_matrix = calcSimMat(
            exp_df=self.exp_set, ref_df=self.ref_set, distance=distance
        )

        hold_out_classes = self.exp_set.sample(frac=0.1, replace=False, axis="index")
        hold_out_classes = {k: [] for k in hold_out_classes.index}

        ref_classes = {
            c: {"comps": []}
            for c in self.exp_set.index.get_level_values(self.classes_col).unique()
        }
        for name, df in self.exp_set.groupby(level=self.classes_col):
            ref_classes[name]["comps"] = list(df.index.get_level_values(0).values)

        for held_comp in hold_out_classes.keys():
            hold_out_classes[held_comp] = {k: [] for k in ref_classes.keys()}
            for c, v in ref_classes.items():
                held_comp_mean2Class = np.nanmean(
                    sim_matrix.loc[held_comp, v["comps"]].values
                )
                kde_support, kde_pdf = self.kde_dict[c]
                held_comp_integral = np.trapz(
                    y=kde_pdf[np.where(kde_support < held_comp_mean2Class)],
                    x=kde_support[np.where(kde_support < held_comp_mean2Class)],
                )
                held_comp_eval = held_comp_integral * len(ref_classes.keys())

                hold_out_classes[held_comp][c] = {
                    "avg_dist": held_comp_mean2Class,
                    "p-val": held_comp_integral,
                    "e-val": held_comp_eval,
                }

        print(
            json.dumps({k[0]: v for k, v in hold_out_classes.items()}, indent=4),
            file=sys.stdout,
        )

        # j_str = json.dumps(ref_classes, indent=4)
        # print(j_str)
        # res_dict = {
        #     "class": [],
        #     "compound": [],
        #     "avg_dist_score": [],
        #     "integral": [],
        #     "e-val": [],
        # }
        # for k, v in ref_classes.items():
        #     if len(v["comps"]) > 0:
        #         # for k in ref_classes.keys(): another loop to test all classes
        #         for comp in v["comps"]:
        #             c_comp_mean2Class = np.nanmean(
        #                 sim_matrix.loc[
        #                     comp, v["comps"]
        #                 ].values  # change this to call cols in row
        #             )

        #             kde_support, kde_pdf = self.kde_dict[k]
        #             c_comp_integral = np.trapz(
        #                 y=kde_pdf[np.where(kde_support < c_comp_mean2Class)],
        #                 x=kde_support[np.where(kde_support < c_comp_mean2Class)],
        #             )
        #             e_val = c_comp_integral * len(ref_classes.keys())

        #             res_dict["class"].append(k)
        #             res_dict["compound"].append(comp)
        #             res_dict["avg_dist_score"].append(c_comp_mean2Class)
        #             res_dict["integral"].append(c_comp_integral)
        #             res_dict["e-val"].append(e_val)
        # res_df = pd.DataFrame.from_dict(res_dict)
        # return res_df


######### VALIDATION

# Random sample from og ref_set (10%) -> This will be out exp set
# Do the above n times

# call run on the above
