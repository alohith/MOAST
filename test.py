#!/usr/bin/env python
import pandas as pd, numpy as np, joblib
from MOAST import Build, GenSimMats, Run
import pyarrow as pa
import os, sys


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

    # res = run._merge_annots(df=run.exp_set)
    # print(res.shape, testData.shape)
    # print(*res.index.get_level_values("AL_CONSOLIDATED"), sep="\n")
    res_df = run.run()
    # res_df.to_csv("test_out.csv")


if __name__ == "__main__":
    main()
