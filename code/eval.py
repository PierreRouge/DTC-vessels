#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:30:12 2023

@author: rouge
"""

import csv
from glob import glob
from tqdm import tqdm

import numpy as np
import nibabel as nib

from skimage.morphology import skeletonize


# Metrics

def dice_numpy(y_true, y_pred):
    """
    Compute dice on numpy array

    Parameters
    ----------
    y_true : Numpy array of size (dim1, dim2, dim3)
        Ground truth
    y_pred : Numpy array of size (dim1, dim2, dim3)
         Predicted segmentation

    Returns
    -------
    dice : Float
        Value of dice

    """
    epsilon = 1e-5
    numerator = 2 * (np.sum(y_true * y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice


def tprec_numpy(y_true, y_pred):
    epsilon = 1e-5
    y_pred = skeletonize(y_pred) / 255
    numerator = np.sum(y_true * y_pred)
    denominator = np.sum(y_pred)
    tprec = (numerator + epsilon) / (denominator + epsilon)
    return tprec


def tsens_numpy(y_true, y_pred):
    epsilon = 1e-5
    y_true = skeletonize(y_true) / 255
    numerator = (np.sum(y_true * y_pred))
    denominator = np.sum(y_true)
    tsens = (numerator + epsilon) / (denominator + epsilon)
    return tsens


def cldice_numpy(y_true, y_pred):
    tprec = tprec_numpy(y_true, y_pred)
    tsens = tsens_numpy(y_true, y_pred)
    numerator = 2 * tprec * tsens
    denominator = tprec + tsens
    cldice = numerator / denominator
    return cldice


def sensitivity_specificity_precision(y_true, y_pred):
    tp = np.sum(y_pred * y_true)
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    sens = tp / np.sum(y_true)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    
    return sens, spec, prec


dir_inputs = '../model/DTC_mse_lr=0.001_cons=1.0_beta=0.3_34labels_beta_0.3/test/*_pred.nii.gz'

listt = glob(dir_inputs)

res = open(dir_inputs.replace('/*_pred.nii.gz', '') + '/../res.csv', 'w')
fieldnames = ['Patient', 'Dice', 'clDice', 'Precision', 'Sensitivity']
writer = csv.DictWriter(res, fieldnames=fieldnames)
writer.writeheader()

dice_list = []
cldice_list = []
prec_list = []
sens_list = []
for item in tqdm(listt):
    pred = nib.load(item)
    gt = nib.load(item.replace('_pred', '_gt'))
    name = item.split('/')[-1]
    name = name.split('_')[0]
    
    pred = pred.get_fdata()
    gt = gt.get_fdata()
    
    dice = dice_numpy(gt, pred)
    cldice = cldice_numpy(gt, pred)
    sens, spec, prec = sensitivity_specificity_precision(gt, pred)
    
    dice_list.append(dice)
    cldice_list.append(cldice)
    prec_list.append(prec)
    sens_list.append(sens)
    
    dict_csv = {'Patient': name,
                'Dice': dice,
                'clDice': cldice,
                'Precision': prec,
                'Sensitivity': sens}
    
    writer.writerow(dict_csv)
    
    print()
    print()
    print(f"Dice:{dice}")
   
print()
print('Moyenne Dice')
print(np.mean(dice_list)) 
print('Moyenne clDice')
print(np.mean(cldice_list))
print('Precision Mean')
print(np.mean(prec_list))
print('Sensitiviy Mean')
print(np.mean(sens_list))
res.close()

    
    
    
    