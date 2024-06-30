#!/usr/bin/env python
import pandas as pd, numpy as np, joblib
from MOAST import Build, GenSimMats, Run
import pyarrow as pa
import os, sys, json


# def main():
#     testdf = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/FeatureReducedHD/TM_1-27_1+10_full_0.8_horiztacked.csv"
#     annots = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Annotations/reducedKey_cytoscapeAnnot.xlsx"

#     # testdf = "/Users/dterciano/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uM_concats_complete/TargetMol_10uM_NoPMA_plateConcat_HD.csv"
#     # annots = "/Users/dterciano/Desktop/LokeyLabFiles/TargetMol/Annotations/TM_GPT-4_Annots_final.csv"

#     annots = pd.read_excel(annots)
#     testData = pd.read_csv(testdf, index_col=0, engine="pyarrow")
#     testData.index.name = "IDname"
#     print(testData.index, annots)
#     testNullData = testData.copy()

#     def renameX(x):
#         strSplit = x.split("._.")[0]
#         strSplit, _, _ = strSplit.rpartition("_")
#         return strSplit

#     testNullData.rename(index=lambda x: renameX(x), inplace=True)

#     b = Build(
#         dataset=testData,
#         # nullData=testNullData,
#         classesDf=annots,
#         on="IDname",
#         classesCol="AL_CONSOLIDATED",
#     )
#     # b.to_pickle("test.csv.pkl")
#     refDist = b.build()
#     print(refDist)
#     b.to_pickle("test.csv.pkl.gzip")
#     joblib.dump(refDist, "kdeDict.pkl.gz", compress="gzip")
#     print(f"Generated Null Data", file=sys.stderr)

#     # refDf = b.getRefDist

#     # g = GenSimMats(expData=testData, refData=refDf.T)
#     # res = g.getReportDf
#     # print(res)

#     # res.to_pickle("simMat.csv.pkl.gzip", compression="gzip")


##### starting from pickle point
def main():
    testdf = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/FeatureReducedHD/TM_1-27_1+10_full_0.8_horiztacked.csv"
    annots = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Annotations/reducedKey_cytoscapeAnnot.xlsx"
    picklFile = "/home/derfelt/gitRepos/MOAST/test.csv.pkl.gzip"

    # refDf = pd.read_pickle(picklFile, compression="gzip")
    annots = pd.read_excel(annots)
    ref_set = pd.read_csv(testdf, index_col=0)
    ref_set.index.name = "id"

    exp_set = ref_set.copy()
    # class_counts = exp_set.index.get_level_values("AL_CONSOLIDATED")
    # print(testData)
    # print(testData.index, refDf.shape)
    # print(annots)

    # testNullData = testData.copy()

    kde_dict = joblib.load("kdeDict.pkl.gz")
    # for k, v in kde_dict.items():
    #     print(k)
    #     print(v)
    #     print(len(v))
    #     break

    # print(testData.shape)
    run = Run(
        exp_set=exp_set,
        ref_set=ref_set,
        kde_dict=kde_dict,
        annots=annots,
        on="IDname",
        classes_col="AL_CONSOLIDATED",
    )

    res_dict, res_df = run.run()

    res_df.to_csv("1_holdout_raw.csv")

    with open("out.json", "w") as f:
        json.dump({str(k): v for k, v in res_dict.items()}, f, indent=4)


##### starting after running calc
# def main():
#     json_path = "/home/derfelt/gitRepos/MOAST/out.json"
#     with open(json_path, "r") as f:
#         out_dict = json.load(
#             f,
#         )

#     res_dict = {
#         "held_comp": [],
#         "tested_class": [],
#         "avg_dist": [],
#         "p-val": [],
#         "e-val": [],
#     }
#     for held_comp, class_dict in out_dict.items():
#         for class_test, res in class_dict.items():
#             res_dict["held_comp"].append(held_comp)
#             res_dict["tested_class"].append(class_test)
#             res_dict["avg_dist"].append(res["avg_dist"])
#             res_dict["p-val"].append(res["p-val"])
#             res_dict["e-val"].append(res["e-val"])

#     # print(json.dumps(res_dict, indent=4))
#     res_df = pd.DataFrame.from_dict(res_dict, orient="columns")
#     # res_df.to_csv("1_holdout.csv", index=False)

#     df = res_df.copy()
#     pivot_df = df.pivot(index="held_comp", columns="tested_class", values="avg_dist")
#     annots = pd.read_excel(
#         "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Annotations/reducedKey_cytoscapeAnnot.xlsx",
#         sheet_name="reducedKey",
#     )
#     annots_pert = annots.loc[
#         annots["IDname"].isin(pivot_df.index), ["IDname", "AL_CONSOLIDATED"]
#     ].copy()
#     pivot_df.reset_index(inplace=True)
#     merged_df = pd.merge(
#         left=pivot_df, right=annots_pert, left_on="held_comp", right_on="IDname"
#     )
#     merged_df.set_index("held_comp", inplace=True)
#     merged_df.drop(columns="IDname", inplace=True)
#     merged_df = merged_df.groupby("AL_CONSOLIDATED").agg("mean")
#     print(annots_pert)


if __name__ == "__main__":
    main()
