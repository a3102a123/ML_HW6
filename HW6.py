from cv2 import threshold
import numpy as np 
import matplotlib.pyplot as plt 
import math
import os
import sys
import time

is_test = True
is_newfile = True
save_folder = "Matrix_s"
new_save_folder = "Matrix_s"

theta = [1,1]
thresholding = 0.001
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
    os.makedirs(new_save_folder,exist_ok = True)
    np.save(file_path,matrix)

# k-means algorithm
### data is a 2-D array. The rows are the data points , the columns are features
def init_center(data,k):
    centers = np.zeros((k,data.shape[1]))
    feature_mean = np.mean(data,axis=0)
    feature_std = np.std(data,axis=0)
    for c in range(data.shape[1]):
        centers[:,c]=np.random.normal(feature_mean[c],feature_std[c],size=k)
    return centers

def k_means(data,K):
    result = np.zeros(len(data),dtype=np.uint8)
    centers = init_center(data,K)
    iteration = 1
    while True:
        # E-step (clustering the data points by closest center)
        for i,d in enumerate(data):
            min_dis = sys.float_info.max
            for l in range(K):
                dis = np.linalg.norm(d - centers[l],2)
                if dis < min_dis:
                    min_dis = dis
                    result[i] = l  
        # M-step
        diff = 0
        for l in range(K):
            idx = (result==l)
            if idx.sum() != 0:
                new_center = np.mean(data[idx],axis=0)
            else:
                new_center = np.array([0,0,0])
            diff += np.linalg.norm(new_center - centers[l],2)
            centers[l] = new_center
        print("K-means iteration : {} , difference : {}".format(iteration,diff))
        draw_label(result,img1)
        plt.show()
        iteration += 1
        if diff <= thresholding:
            break

    return result

# Default assume label coming from k-means. It's 1-D array
def draw_label(label,img):
    h,w,c = img.shape
    picture = np.zeros((h,w))
    for i,l in enumerate(label):
        idx = get_pos(i,w)
        picture[idx[0],idx[1]] = l
    plt.figure()
    plt.imshow(picture)

# Calc similarity graphy
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

def degree_matrix(W):
    D = np.zeros_like(W)
    h,w = D.shape
    for i in range(h):
        sum = 0
        for j in range(w):
            sum += W[i,j]
        D[i,i] = sum
    return D 

# sepctral clustering
def spectral():
    # prepare associated matrix W & D
    if is_newfile and not os.path.exists(new_save_folder):
        W = weighted_graph(img1)
        D = degree_matrix(W)
        save_matrix(new_save_folder,"W",W)
        save_matrix(new_save_folder,"D",D)
    else:
        W = np.load(os.path.join(save_folder,"W.npy"))
        D = np.load(os.path.join(save_folder,"D.npy"))
        print("Pre-computed similarity matrix (W) and degree matrix (D) already exist!")
    
    L = D - W
    # calc the eigenvalue & eigenvector of Laplacian graph
    eigenValues , eigenVectors = np.linalg.eig(L)
    sorted_eigen_idx = np.argsort(eigenValues)
    for i in range(20):
        print(eigenValues[sorted_eigen_idx[i]],sorted_eigen_idx[i]) 

# print(kernel(np.array([1,1]),np.array([50,1]),np.array([1,1,1]),np.array([1,1,1]),theta))

# main
if is_test:
    # down sampling to create a small image
    img1 = img1[::5,::5,:]
h,w,c = img1.shape
img1_data = img1.reshape((h*w,c))
label = k_means(img1_data,4)
# spectral()
plt.show()