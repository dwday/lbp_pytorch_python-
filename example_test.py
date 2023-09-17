#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance evaluation of lbp using pytorch

"""
from lib.lbplib import lbp_py,lbp_pt
import numpy as np
import time

import torch
  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("torch.cuda.is_available():",torch.cuda.is_available())


# Test data with random numbers
Rows=128
Cols=128
img_org=np.random.randint(0,255,(Rows,Cols))

 
[Rows,Cols]=img_org.shape
img_lbp=img_org.reshape(1,Rows,Cols)

# Test Python implementation 
start_time   = time.time()    
ypy  = lbp_py(img_lbp[0,:,:].astype('uint8'))
elapsed_py = time.time() - start_time    
print('python -  elapsed_time  =',elapsed_py)

# Test pytorch implementation
start_time   = time.time()


img_torch =torch.from_numpy(img_lbp) 
ypt = lbp_pt(img_torch ).numpy()

elapsed_pt = time.time() - start_time
print('pytorch - elapsed_time  =',elapsed_pt)

#Check if there is an error between PyTorch and Python implementations
print('error=',np.sum(ypt[0,:,:]-ypy))

print('python:')
print(ypy)
print('PyTorch:')
print(ypt[0,:,:])    