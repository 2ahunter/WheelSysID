#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:56:30 2019

@author: aahunter
"""
import numpy as np
from scipy.interpolate import CubicSpline

def calcTension(d):
    d_cal = np.array([0.58,0.87,1.09,1.18,1.27,1.35,1.43,1.5,1.56,1.62,1.68,1.73,1.78,1.83,1.88,1.92,1.97,2.05,2.12,2.19,2.25,2.31,2.38])
    T_cal = np.array([300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1600, 1700, 1800])
    CS = CubicSpline(d_cal,T_cal)
    T = CS(d)
    return T