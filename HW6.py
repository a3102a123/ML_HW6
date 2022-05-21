import numpy as np 
import matplotlib.pyplot as plt 
import math
import os
import sys
import time
import imageio
import cv2

is_test = True
is_newfile = False
save_folder = "Matrix_s"
new_save_folder = "Matrix_s"

theta = [0.0007,0.0007]
thresholding = 0.01
img1 = plt.imread("image1.png")
img2 = plt.imread("image2.png")

# gif class
label_to_color = {
    0: [255,  0, 0],
    1: [0,  255, 0],
    2: [ 0,  0,  255],
    3: [255, 255, 0],
    4: [0, 255, 255],
    5: [255, 0, 255],
    6: [255,255,255]
}
class gif_creater:
    def __init__(self,fps=0.8):
        self.imgs = []
        self.fps = fps

    def append(self,img):
        self.imgs.append(img)
    
    def append_label(self,label):
        img = self.label2Image(label)
        self.append(img)
        
    def label2Image(self,label):
        img = np.zeros((label.shape[0],label.shape[1],3),dtype=np.uint8)
        for l,color in label_to_color.items():
            img[label == l,:] = color
        return img

    def save(self,filepath):
        self.ToInt()
        self.resize(400)
        imageio.mimsave(filepath, self.imgs, fps=self.fps)

    def clear(self):
        self.imgs.clear()

    def resize(self,size):
        if self.imgs[0].shape[0] != size:
            for i,img in enumerate(self.imgs):
                self.imgs[i] = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)

    def ToInt(self):
        for i,img in enumerate(self.imgs):
            if not np.issubdtype(img.dtype,np.uint8):
                self.imgs[i] = (img * 255).astype(np.uint8)

gif = gif_creater()

# kernel function
def kernel(a_pos,b_pos,a_c,b_c,theta):
    spatial = np.exp(-theta[0] * np.linalg.norm((a_pos - b_pos),2) ** 2)
    color = np.exp(-theta[1] * np.linalg.norm((a_c - b_c),2) ** 2)
    return spatial * color

def get_pos(idx,width):
    i = math.floor(idx / width)
    j = idx % width
    return np.array([i,j])

def save_Matrix(folder_name,file_name,matrix):
    file_path = os.path.join(folder_name,file_name)
    os.makedirs(new_save_folder,exist_ok = True)
    np.save(file_path,matrix)

# k-means algorithm
### data is a 2-D array. The rows are the data points , the columns are features
def init_center_random(data,K):
    centers = np.zeros((K,data.shape[1]))
    feature_mean = np.mean(data,axis=0)
    feature_std = np.std(data,axis=0)
    for c in range(data.shape[1]):
        centers[:,c]=np.random.normal(feature_mean[c],feature_std[c],size=K)
    return centers

def init_center_plus(data,K):
    centers = np.zeros((K,data.shape[1]))
    centers_idx = np.zeros((K))
    idx = np.random.randint(0,data.shape[0],1)
    centers_idx[0] = idx
    centers[0,:] = data[idx,:]
    center_num = 1
    while center_num < K:
        dis = np.zeros(data.shape[0])
        for i,d in enumerate(data):
            for j in range(center_num):
                temp_dis = np.linalg.norm(centers[j] - d,2)
                if j == 0:
                    dis[i] = temp_dis
                else:
                    dis[i] = min(dis[i],temp_dis)
        dis = dis / dis.sum()
        idx = np.argmax(dis)
        centers[center_num,:] = data[idx,:]
        centers_idx[center_num] = idx
        center_num +=1
    print(centers_idx)
    return centers

def k_means(img,data,K):
    result = np.zeros(len(data),dtype=np.uint8)
    centers = init_center_plus(data,K)
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
        # M-step (calculating the new center point)
        diff = 0
        for l in range(K):
            idx = (result==l)
            if idx.sum() != 0:
                new_center = np.mean(data[idx],axis=0)
            else:
                new_center = np.zeros(data.shape[1])
            diff += np.linalg.norm(new_center - centers[l],2)
            centers[l] = new_center
        print("K-means iteration : {} , difference : {}".format(iteration,diff))
        draw_label(result,img)
        # plt.show()
        iteration += 1
        if diff <= thresholding:
            break

    return result

# kernel k means
def kernel_k_means(img,W,K):
    N = W.shape[0]
    result = np.random.randint(0,K,N)
    pre_result = np.zeros(N)
    pre_result.fill(-1)
    dist = np.zeros((N,K))
    iteration = 1
    draw_label(result,img)
    while True:
        dist.fill(0)
        for c in range(K):
            mask = (result == c)
            c_num = np.sum(mask)
            dist[:,c] = np.diag(W)
            KK = W[mask][:,mask]
            dist[:,c] += np.sum(KK)/(c_num**2)
            dist[:,c] -= 2*np.sum(W[:,mask],axis=1)/c_num
            # print(c_num,dist[:,c].shape,W[:,mask].shape,KK.shape)
        pre_result = result.copy()
        result = dist.argmin(axis=1)
        
        draw_label(result,img)
        change_num_ratio = 1 - np.sum((pre_result - result) == 0) / N
        print("K-means iteration : {} , The ratio of changing label : {}".format(iteration,change_num_ratio))
        iteration += 1
        if  change_num_ratio < thresholding:
            break
    
    return result

# Default assume label coming from k-means. It's 1-D array
def draw_label(label,img):
    h,w,c = img.shape
    label_picture = np.zeros((h,w))
    for i,l in enumerate(label):
        idx = get_pos(i,w)
        label_picture[idx[0],idx[1]] = l
    gif.append_label(label_picture)
    plt.figure()
    plt.imshow(label_picture)

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

# prepare associated matrix W & D
# W is similarity matrix
# D is degree matrix
def prepare_Matrix(W_fileName , D_fileName , img):
    strat_time = time.time()

    W = weighted_graph(img)
    D = degree_matrix(W)
    save_Matrix(new_save_folder,W_fileName,W)
    save_Matrix(new_save_folder,D_fileName,D)
        
    end_time = time.time()
    time_c= end_time - strat_time
    min_c = int(time_c / 60)
    time_c = time_c - min_c * 60
    print('Preparing data total time cost : {}m , {:.3f}s'.format(min_c,time_c))
    return W,D

# load pre-computing matrix
def load_Matrix(W_fileName , D_fileName):
    W = np.load(os.path.join(save_folder,"{}.npy".format(W_fileName)))
    D = np.load(os.path.join(save_folder,"{}.npy".format(D_fileName)))
    return W,D

# sepctral clustering
def spectral(K,W,D,is_norm):
    L = D - W
    # normalized Laplacian
    if is_norm:
        D_inv_srt = np.diag(1/np.diag(np.sqrt(D)))
        L = D_inv_srt @ L @ D_inv_srt
        
    # calc the eigenvalue & eigenvector of Laplacian graph
    eigenValues , eigenVectors = np.linalg.eig(L)
    sorted_eigen_idx = np.argsort(eigenValues)
    print(eigenVectors.shape)
    H = eigenVectors[:,sorted_eigen_idx[1:K+1]]

    # normalize the norm of every row to 1
    if is_norm:
        sum = np.linalg.norm(H,axis=1)
        H = H / sum.reshape(-1,1)

    print(np.linalg.norm(H[0,:]))
    label = k_means(img1,H,K)

# print(kernel(np.array([1,1]),np.array([50,1]),np.array([1,1,1]),np.array([1,1,1]),theta))

# main
if is_test:
    # down sampling to create a small image
    img1 = img1[::5,::5,:]
h,w,c = img1.shape
img1_data = img1.reshape((h*w,c))
if is_newfile or not os.path.exists(new_save_folder):
    W1,D1 = prepare_Matrix("W","D",img1)
else:
    print("Pre-computed similarity matrix (W) and degree matrix (D) already exist!")
    W1,D1 = load_Matrix("W","D")
# label = k_means(img1,img1_data,4)
# label = kernel_k_means(img1,W1,4)
gif.append(img1)
spectral(4,W1,D1,True)
plt.figure()
plt.imshow(img1)
gif.save("Result/result.gif")
plt.show()