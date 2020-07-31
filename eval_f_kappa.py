# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:28:40 2020

@author: de_hauk
"""
import pandas as pd
from f_kappa_public import f_kappa_new_boot



path_private = r'data_FQ_2015.xlsx'
data_raw = pd.read_excel(path_private)
data = data_raw.drop(labels=['N_raters','Subject'], axis=1)

res = f_kappa_new_boot(df = data,raters=6,iter_permute=100,iter_boot=1000)