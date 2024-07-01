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

    def run(self, distance: bool = True) -> tuple[dict, pd.DataFrame]:
        sim_matrix = calcSimMat(
            exp_df=self.exp_set, ref_df=self.ref_set, distance=distance
        )

        hold_out_classes = self.exp_set.sample(
            frac=0.1, replace=False, axis="index", random_state=15
        )
        hold_out_classes = {k: [] for k in hold_out_classes.index}

        ref_classes = {
            c: {"comps": []}
            for c in self.exp_set.index.get_level_values(self.classes_col).unique()
        }
        for name, df in self.exp_set.groupby(level=self.classes_col):
            ref_classes[name]["comps"] = list(df.index.get_level_values(0).values)

        for held_comp in hold_out_classes.keys():
            print(held_comp, file=sys.stderr)
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

        res_dict = {
            "held_comp": [],
            "held_class": [],
            "tested_class": [],
            "avg_dist": [],
            "p-val": [],
            "e-val": [],
        }
        for (held_comp, held_class), class_dict in hold_out_classes.items():
            for class_test, res_info in class_dict.items():
                res_dict["held_comp"].append(held_comp)
                res_dict["held_class"].append(held_class)
                res_dict["tested_class"].append(class_test)
                res_dict["avg_dist"].append(res_info["avg_dist"])
                res_dict["p-val"].append(res_info["p-val"])
                res_dict["e-val"].append(res_info["e-val"])

        return hold_out_classes, pd.DataFrame.from_dict(res_dict, orient="columns")


def square_form(
    holdout_csv: pd.DataFrame, p_val=False, kde_dict: dict = None
) -> pd.DataFrame:
    from . import MoastException

    piv_df = holdout_csv.pivot(
        index=["held_comp", "held_class"], columns="tested_class", values="avg_dist"
    ).reset_index("held_comp", drop=True)
    return_df = piv_df.groupby(level="held_class").agg("mean")

    if p_val:
        if kde_dict is None:
            raise MoastException("a kde_dict needs to be provided")

        def integrate_KDE(cl_name: str, avg_dists: np.array, kde_dict: dict):
            kde_support, kde_pdf = kde_dict[cl_name]
            res = []
            for x in avg_dists:
                integral = np.trapz(
                    y=kde_pdf[np.where(kde_support < x)],
                    x=kde_support[np.where(kde_support < x)],
                )
                res.append(integral)
            return res

        return_df = return_df.apply(
            lambda x: integrate_KDE(cl_name=x.name, avg_dists=x, kde_dict=kde_dict)
        )
    return return_df

    # print(
    #     json.dumps({k[0]: v for k, v in hold_out_classes.items()}, indent=4),
    #     file=sys.stdout,
    # )

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
