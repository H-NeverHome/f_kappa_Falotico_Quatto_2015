# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:28:45 2020

@author: hauke
"""


def f_kappa_new_boot(df,raters,iter_permute,iter_boot):
    from sklearn.utils import shuffle
    import pandas as pd
    import numpy as np
    from arch.bootstrap import IIDBootstrap
    df = df
    raters = raters
    ########################################
    # Define Fuctions
    
    def binom_coeff(x,raters):
        return (x*(x-1))/ (raters*(raters-1))
    
    def permute_rows(row):
       row_raw = [i for i in row]
       row_1 = shuffle(row_raw)
       return pd.Series(row_1)
   
    def f_kappa_old(df,raters):
        raters = raters
        n_subj= df.shape[0]
        #n_categorization = df.sum(axis = 1)
        data1 = df.apply(binom_coeff, axis=1,raters=raters)
        p_i = data1.sum(axis = 1)
        p_bar = p_i.sum()/len(p_i) #overall_agreement 
        
        p_jj = df.sum(axis=0)/(n_subj*raters)
        p_bar_e = (p_jj**2).sum()
        fleiss_kappa = (p_bar-p_bar_e)/(1-p_bar_e)
        return fleiss_kappa
    
    def f_kappa_new(df,raters,iterations):
        res_kapp=[]
        for i in range(iterations):
            data_shuffle = df.apply(permute_rows, axis=1)
            f_kapp = f_kappa_old(data_shuffle,6)
            res_kapp.append(f_kapp)  
        return np.median(res_kapp)      
    
    ########################################
    # calc normal fleiss_kappa
    f_kappa_normal = f_kappa_old(df,6)
    
    ########################################
    # calc robust fleiss_kappa with permutation
    res_kappa= []
    for i in range(iter_permute):
        data_shuffle = df.apply(permute_rows, axis=1)
        f_kapp = f_kappa_old(data_shuffle,6)
        res_kappa.append(f_kapp)  
    f_kappa_perm =  np.median(res_kappa) 
    ########################################
    # calc robust fleiss_kappa with permutation AND bootstrapping
    bs = IIDBootstrap(df)
    ci = bs.conf_int(f_kappa_new, reps = iter_boot, method='bca',extra_kwargs = {'raters':raters,'iterations' :iter_permute})

    return {'iterations_permute':iter_permute,
            'iterations_bootstrapped':iter_boot,
            'data': df,
            'unperm_f_kappa':f_kappa_normal,
            'perm_f_kappa' : f_kappa_perm,
            'ci':ci}
