#!/usr/bin/env python

import pandas as pd, numpy as np
import os, sys
from .buildingStage import Build
from .genSimMats import GenSimMats

####### TODO: MOAST GOES HERE #######


class MoastException(Exception):
    def __init__(self, message, *args: object) -> None:
        super().__init__(message, *args)


# global steps:
# 1) build (null model)
# 2) featRport (gen sim rport)
# 3) intergrate/predict (option to do ks test: dist query of 2 members of class vs. dist of all other members)
