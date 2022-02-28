# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 10:55:11 2020

@author: bilg
"""
import torch
#import os
import torch.nn.functional as F
#import cv2
import numpy as np
#import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("torch.cuda.is_available():",torch.cuda.is_available())
#------------------------------------------------------------------------------
def lbp_python(Im): 
    
    sat=len(Im)
    sut=len(Im[0])
    L=np.zeros((sat,sut))
    I=np.zeros((sat+2,sut+2)) 
    I[1:sat+1,1:sut+1]=Im
    for i in range(1,sat+1):
        for j in range(1,sut+1):
            L[i-1,j-1]=\
            ( I[i-1,j]  >= I[i,j] )*1+\
            ( I[i-1,j+1]>= I[i,j] )*2+\
            ( I[i,j+1]  >= I[i,j] )*4+\
            ( I[i+1,j+1]>= I[i,j] )*8+\
            ( I[i+1,j]  >= I[i,j] )*16+\
            ( I[i+1,j-1]>= I[i,j] )*32+\
            ( I[i,j-1]  >= I[i,j] )*64+\
            ( I[i-1,j-1]>= I[i,j] )*128;  
    
    return L
#------------------------------------------------------------------------------


def tc_lbp(x):
    #pad image for 3x3 mask size
    x = F.pad(input=x, pad = [1, 1, 1, 1], mode='constant')
    b=x.shape
    M=b[1]
    N=b[2]
    
    y=x
    #select elements within 3x3 mask 
    y00=y[:,0:M-2, 0:N-2]
    y01=y[:,0:M-2, 1:N-1]
    y02=y[:,0:M-2, 2:N  ]
    #     
    y10=y[:,1:M-1, 0:N-2]
    y11=y[:,1:M-1, 1:N-1]
    y12=y[:,1:M-1, 2:N  ]
    #
    y20=y[:,2:M, 0:N-2]
    y21=y[:,2:M, 1:N-1]
    y22=y[:,2:M, 2:N ]      
    
    # Apply comparisons and multiplications 
    bit=torch.ge(y01,y11)
    tmp=torch.mul(bit,torch.tensor(1))  
    
    bit=torch.ge(y02,y11)
    val=torch.mul(bit,torch.tensor(2))
    val=torch.add(val,tmp)    
    
    bit=torch.ge(y12,y11)
    tmp=torch.mul(bit,torch.tensor(4))
    val=torch.add(val,tmp)
    
    bit=torch.ge(y22,y11)
    tmp=torch.mul(bit,torch.tensor(8))   
    val=torch.add(val,tmp)
    
    bit=torch.ge(y21,y11)
    tmp=torch.mul(bit,torch.tensor(16))   
    val=torch.add(val,tmp)
    
    bit=torch.ge(y20,y11)
    tmp=torch.mul(bit,torch.tensor(32))   
    val=torch.add(val,tmp)
    
    bit=torch.ge(y10,y11)
    tmp=torch.mul(bit,torch.tensor(64))   
    val=torch.add(val,tmp)
    
    bit=torch.ge(y00,y11)
    tmp=torch.mul(bit,torch.tensor(128))   
    val=torch.add(val,tmp)    
    return val

print('Random test numbers:')
imgs=np.random.randint(0,256,(7,7))
print(imgs)

# Compute using pytorch
y1=tc_lbp(torch.from_numpy(imgs.reshape(1,7,7)))
# Compute using python
y2=lbp_python(imgs)

print('Python computation result:')
print(y2.reshape(7,7).astype('uint8'))
print('PyTorch computation result:')
print(y1.numpy().reshape(7,7).astype('uint8'))

print('Error check:')
e=y1.numpy()-y2
print(e)

#******************************************************************************


