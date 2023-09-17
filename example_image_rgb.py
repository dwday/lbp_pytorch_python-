
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test file for rgb and grascale outputs  
https://journalengineering.fe.up.pt/index.php/upjeng/article/view/2183-6493_007-004_0005/567
"""
from lib.lbplib import lbp_pt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

# test image
path='img/ILSVRC2012_val_00000328.JPEG'

# Read image
img_rgb =Image.open(path)

#Gray input--------------------------------------------------------------------
img_gray =img_rgb.convert('L')
img_gray=np.asarray(img_gray)
[Rows,Cols]=img_gray.shape
img_torch=torch.from_numpy(img_gray.reshape(1,Rows,Cols).astype('uint8'))
lbp_gray = lbp_pt(img_torch).numpy()


# RGB input -------------------------------------------------------------------
img_rgb=np.asarray(img_rgb)
img_torch0=torch.from_numpy(img_rgb[:,:,0].reshape(1,Rows,Cols).astype('uint8'))
img_torch1=torch.from_numpy(img_rgb[:,:,1].reshape(1,Rows,Cols).astype('uint8'))
img_torch2=torch.from_numpy(img_rgb[:,:,2].reshape(1,Rows,Cols).astype('uint8'))


# Allocation for the rgb output 
lbp_rgb=img_rgb.copy()
lbp_rgb[:,:,0] = lbp_pt(img_torch0).numpy()
lbp_rgb[:,:,1] = lbp_pt(img_torch1).numpy()
lbp_rgb[:,:,2] = lbp_pt(img_torch2).numpy()

# Show input and output images-------------------------------------------------
plt.figure(1)
plt.imshow(img_rgb.astype('uint8'))
plt.title('Input file')    
plt.figure(2)
plt.imshow(lbp_rgb.astype('uint8') )
plt.title('RGB output')
plt.figure(3)
plt.imshow(lbp_gray.reshape(Rows,Cols).astype('uint8') ,cmap='gray' )
plt.title('Grayscale output')