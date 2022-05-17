import numpy as np 
import matplotlib.pyplot as plt 
import math
import os
import sys
import time

is_test = True
is_newfile = True
W_folder = "W_s"
new_W_folder = "W_s"

theta = [1,1]
img1 = plt.imread("image1.png")
img2 = plt.imread("image2.png")

# kernel function
def kernel(a_pos,b_pos,a_c,b_c,theta):
    spatial = np.exp(-theta[0] * np.linalg.norm((a_pos - b_pos),2) ** 2)
    color = np.exp(-theta[1] * np.linalg.norm((a_c - b_c),2) ** 2)
    return spatial * color

def get_pos(idx,width):
    i = math.floor(idx / width)
    j = idx % width
    return np.array([i,j])

def save_matrix(folder_name,file_name,matrix):
    file_path = os.path.join(folder_name,file_name)
    os.makedirs(new_W_folder,exist_ok = True)
    np.save(file_path,matrix)

def weighted_graph(img):
    h,w,c = img.shape
    W = np.zeros((h*w,h*w))
    # weight matrix contains the weight of every two by two pixel
    for i in range(h*w):
        for j in range(h*w):
            i_pos = get_pos(i,w)
            j_pos = get_pos(j,w)
            i_c = img[i_pos[0],i_pos[1],:]
            j_c = img[j_pos[0],j_pos[1],:]
            W[i,j] = kernel(i_pos,j_pos,i_c,j_c,theta)
    return W
            

# sepctral clustering
def spectral():
    if is_newfile and not os.path.exists(new_W_folder):
        W = weighted_graph(img1)
        save_matrix(new_W_folder,"W",W)
    else:
        W = np.load(os.path.join(W_folder,"W.npy"))
        print("Pre-computed similarity matrix (W) already exist!")
    plt.imshow(W)

# print(kernel(np.array([1,1]),np.array([50,1]),np.array([1,1,1]),np.array([1,1,1]),theta))

# main
if is_test:
    # down sampling to create a small image
    img1 = img1[::5,::5,:]
spectral()
plt.show()