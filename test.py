#!/usr/bin/env python
import pandas as pd, numpy as np
from MOAST import Build, GenSimMats
import pyarrow as pa
import os, sys


# def main():
#     testdf = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uM_concats_complete/TargetMol_10uM_NoPMA_plateConcat_HD.csv"
#     annots = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Annotations/TM_GPT-4_Annots_final.csv"

#     # testdf = "/Users/dterciano/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uM_concats_complete/TargetMol_10uM_NoPMA_plateConcat_HD.csv"
#     # annots = "/Users/dterciano/Desktop/LokeyLabFiles/TargetMol/Annotations/TM_GPT-4_Annots_final.csv"

#     annots = pd.read_csv(annots, engine="pyarrow")
#     testData = pd.read_csv(testdf, index_col=0, engine="pyarrow")
#     print(testData.shape)
#     testNullData = testData.copy()

#     def renameX(x):
#         strSplit = x.split("._.")[0]
#         strSplit, _, _ = strSplit.rpartition("_")
#         return strSplit

#     testNullData.rename(index=lambda x: renameX(x), inplace=True)

#     b = Build(
#         dataset=testData,
#         nullData=testNullData,
#         classesDf=annots,
#         on="Name",
#         classesCol="GPT-4 Acronym",
#     )
#     # b.to_pickle("test.csv.pkl")
#     refDist = b.build()
#     b.to_pickle("test.csv.pkl.gzip")
#     print(f"Generated Null Data", file=sys.stderr)

#     refDf = b.getRefDist

#     g = GenSimMats(expData=testData, refData=refDf.T)
#     res = g.getReportDf
#     print(res)

#     res.to_pickle("simMat.csv.pkl.gzip", compression="gzip")


##### starting from pickle point
def main():
    testdf = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uM_concats_complete/TargetMol_10uM_NoPMA_plateConcat_HD.csv"
    annots = "/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Annotations/TM_GPT-4_Annots_final.csv"
    picklFile = "/home/derfelt/gitRepos/MOAST/test.csv.pkl.gzip"

    refDf = pd.read_pickle(picklFile, compression="gzip")
    # annots = pd.read_csv(annots, engine="pyarrow")
    testData = pd.read_csv(testdf, index_col=0, engine="pyarrow")
    # print(testData.shape)
    # testNullData = testData.copy()
    print(testData.shape, refDf.shape, file=sys.stderr)

    # The pickling part is fine, the GenSimMats function is the problem

    # Seg Fault Below
    g = GenSimMats(expData=testData, refData=refDf.T)
    simMat = g.getReportDf

    print(simMat)
    simMat.to_pickle("simMat.csv.pkl")


if __name__ == "__main__":
    main()
