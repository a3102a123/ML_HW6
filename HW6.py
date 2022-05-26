import numpy as np 
import matplotlib.pyplot as plt 
import math
import os
import sys
import time
import imageio
import cv2
from scipy.spatial.distance import *

is_test = True
is_newfile = False
save_folder = "Matrix_s"
new_save_folder = "Matrix_s"

# kernel my way
# theta = [0.00001,0.001]
# kernel faster way
theta = [1,1]
# theta = [1,1]
thresholding = 0.01
img1 = plt.imread("image1.png")
img2 = plt.imread("image2.png")

# gif class
label_to_color = {
    0: [255,  0, 0],
    1: [255,  255, 0],
    2: [ 0,  0,  255],
    3: [0, 255, 0],
    4: [0, 255, 255],
    5: [255, 0, 255],
    6: [255,255,255]
}

def label2Image(label,img):
    h,w,c = img.shape
    label_picture = np.zeros((h,w))
    for i,l in enumerate(label):
        idx = get_pos(i,w)
        label_picture[idx[0],idx[1]] = l

    ret_img = np.zeros((h,w,3),dtype=np.uint8)
    for l,color in label_to_color.items():
        ret_img[label_picture == l,:] = color

    return ret_img

class gif_creater:
    def __init__(self,fps=2):
        self.imgs = []
        self.fps = fps

    def append(self,img):
        file_path = os.path.join("Result","GIF_Image","img{}.png".format(len(self.imgs)))
        plt.figure()
        plt.imshow(img)
        plt.savefig(file_path)
        plt.close()
        img = plt.imread(file_path)
        self.imgs.append(img)
    
    def append_label(self,label):
        img = label2Image(label)
        self.append(img)

    def save(self,filepath):
        self.ToInt()
        # self.resize(400)
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
    spatial = np.exp(-theta[0] * np.linalg.norm((a_pos - b_pos),2) ** 2 / (2*3**2))
    color = np.exp(-theta[1] * np.linalg.norm((a_c - b_c),2) ** 2 / (2*2**2))
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
    return centers

def k_means(img,data,K,is_show=False):
    result = np.zeros(len(data),dtype=np.uint8)
    # k-means++
    centers = init_center_plus(data,K)
    # random
    # centers = init_center_random(data,K)
    iteration = 1
    while True:
        if iteration == 1:
            print(data.shape)
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
        draw_label(result,img,is_show=is_show)
        # plt.show()
        iteration += 1
        if diff <= thresholding and True:
            break

    return result

def init_kernel_center_plus(W,K):
    N = W.shape[0]
    centers = np.zeros(K,dtype=np.uint)
    centers[0] = np.random.randint(0,N,1)
    center_num = 1
    while center_num < K:
        dist = np.zeros(N)
        for i in range(center_num):
            temp_dis = 1 - W[centers[i],:]
            if i == 0:
                dist = temp_dis
            else:
                dist = np.minimum(dist,temp_dis)
        centers[center_num] = np.argmax(dist)
        center_num += 1
    result = np.argmax(W[centers,:],axis=0)
    return result

# kernel k means
def kernel_k_means(img,W,K,is_show=False):
    N = W.shape[0]
    # random initial
    # result = np.random.randint(0,K,N)
    # k-means++
    result = init_kernel_center_plus(W,K)
    
    pre_result = np.zeros(N)
    pre_result.fill(-1)
    dist = np.zeros((N,K))
    iteration = 1
    draw_label(result,img,is_show)
    while True:
        # E-step : calc the distance between data point & cluster center in kernel space
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
        # M-step : assign closest label to data point
        result = dist.argmin(axis=1)
        
        # draw_label(result,img)
        change_num_ratio = 1 - np.sum((pre_result - result) == 0) / N
        print("K-means iteration : {} , The ratio of changing label : {}".format(iteration,change_num_ratio))
        iteration += 1
        if  change_num_ratio < thresholding:
            break
    draw_label(result,img,is_show)
    return result

# Default assume label coming from k-means. It's 1-D array
def draw_label(label,img,is_show=False):
    label_img = label2Image(label,img)
    gif.append(label_img)
    if is_show:
        plt.figure()
        plt.imshow(label_img)

# Calc similarity graphy
def weighted_graph(filename,img):
    # the faster way finded on internet
    h,w,c = img.shape
    img_data = img.reshape((h*w,c))
    n=len(img_data)
    spacial_idx=np.zeros((n,2))
    for i in range(n):
        spacial_idx[i]=[i//h,i%h]
    spacial = squareform(pdist(spacial_idx,'sqeuclidean'))
    color = squareform(pdist(img_data,'sqeuclidean'))
    save_Matrix(new_save_folder,"spacial_{}".format(filename),spacial)
    save_Matrix(new_save_folder,"color_{}".format(filename),color)
    spacial = np.exp(-theta[0]*spacial)
    color = np.exp(-theta[1]*color)
    W = spacial * color
    return W

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
    D = np.diag(np.sum(W,axis=1))
    return D
    D = np.zeros_like(W)
    h,w = D.shape
    for i in range(h):
        sum = 0
        for j in range(w):
            sum += W[i,j]
        D[i,i] = sum
    return D 

# using pre-computed spacial & color kernel with hyper-parameter to combine different similarity matrix & degree matrix
def kernel2Matrix(spacial,color,theta):
    spacial = np.exp(-theta[0]*spacial)
    color = np.exp(-theta[1]*color)
    W = spacial * color
    D = degree_matrix(W)
    return W,D

# prepare associated matrix W & D
# W is similarity matrix
# D is degree matrix
def prepare_Matrix(filename , img):
    start_time = time.time()

    W = weighted_graph(filename,img)
    D = degree_matrix(W)
    save_Matrix(new_save_folder,"W_{}".format(filename),W)
    save_Matrix(new_save_folder,"D_{}".format(filename),D)
        
    end_time = time.time()
    time_c= end_time - start_time
    min_c = int(time_c / 60)
    time_c = time_c - min_c * 60
    print('Preparing data total time cost : {}m , {:.3f}s'.format(min_c,time_c))

# load pre-computing matrix
def load_Matrix(filename):
    W = np.load(os.path.join(save_folder,"W_{}.npy".format(filename)))
    D = np.load(os.path.join(save_folder,"D_{}.npy".format(filename)))
    spacial = np.load(os.path.join(save_folder,"spacial_{}.npy".format(filename)))
    color = np.load(os.path.join(save_folder,"color_{}.npy".format(filename)))
    return W,D,spacial,color

# sepctral clustering
def spectral(K,W,D,is_norm,img,is_show_H = False,is_show = False):
    L = D - W
    # normalized Laplacian
    if is_norm:
        D_inv_srt = np.diag(1/np.diag(np.sqrt(D)))
        L = D_inv_srt @ L @ D_inv_srt
        
    # calc the eigenvalue & eigenvector of Laplacian graph
    eigenValues , eigenVectors = np.linalg.eig(L)
    sorted_eigen_idx = np.argsort(eigenValues)
    # let smallest eigenvector = 1 vector
    # eigenVectors /= eigenVectors[:,sorted_eigen_idx[0]].reshape(-1,1)
    H = eigenVectors[:,sorted_eigen_idx[1:K+1]]

    # normalize the norm of every row to 1
    if is_norm or True:
        sum = np.linalg.norm(H,axis=1)
        H = H / sum.reshape(-1,1)

    label = k_means(img,H,K,is_show)

    # visualizing H
    if is_show_H:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(H[:,0], H[:,1], H[:,2])
        H = H.reshape(img.shape).astype(np.float64)
        for i in range(3):
            H[:,:,i] = (H[:,:,i] - np.min(H[:,:,i])) / (np.max(H[:,:,i]) - np.min(H[:,:,i]))
        plt.figure()
        plt.imshow(H)
    
    return label

# calc kernel without hyperparameter for grid search
def kernel_for_search(img):
    # the faster way finded on internet
    h,w,c = img.shape
    img_data = img.reshape((h*w,c))
    n=len(img_data)
    spacial_idx=np.zeros((n,2))
    for i in range(n):
        spacial_idx[i]=[i//h,i%h]
    spacial = squareform(pdist(spacial_idx,'sqeuclidean'))
    color = squareform(pdist(img_data,'sqeuclidean'))
    return spacial,color

# mode 0 : kernel k-means , 1 : ratio cut , 2 : normalized cut
def grid_search(K,img,mode):
    name_dict = {0: "Kernel K-means" , 1: "Ratio Cut" , 2: "Normalized cut"}
    #theta 0 for spacial , 1 for color
    theta = [1,1]
    fig = plt.figure()
    gs = fig.add_gridspec(5, 5, hspace=0, wspace=0)
    fig.suptitle("{} Hyperparameter grid search".format(name_dict[mode]),fontsize = 20,y=0.92)
    fig.text(0.5, 0.07, "Color theta", ha='center',fontsize = 14)
    fig.text(0.07, 0.5, "Spacial theta", va='center', rotation='vertical',fontsize = 14)
    axs = gs.subplots(sharex='col', sharey='row')
    fig.set_figheight(9)
    fig.set_figwidth(9)

    k_spacial,k_color = kernel_for_search(img)
    for i in range(5):
        for j in range(5):
            ax = axs[i,j]
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            # ax.axis("off")
            theta_s = -theta[0] * min(1,(1 / 10 ** (i+1)))
            theta_c = -theta[1] * min(1,(1 / 10 ** j))
            if j == 0:
                ax.set(ylabel=theta_s)
            if i == 4:
                ax.set(xlabel=theta_c)
            W = np.exp(theta_s*k_spacial) * np.exp(theta_c*k_color)
            D = degree_matrix(W)
            if mode == 0:
                label = kernel_k_means(img,W,K)
            elif mode == 1:
                label = spectral(K,W,D,False,img)
            elif mode == 2:
                label = spectral(K,W,D,True,img)
            ax.imshow(label2Image(label,img))
            ax.axis('tight')
            # ax.plot(W)

# print(kernel(np.array([1,1]),np.array([50,1]),np.array([1,1,1]),np.array([1,1,1]),theta))

# main
if is_test:
    # down sampling to create a small image
    img1 = img1[::5,::5,:]
h,w,c = img1.shape
img1_data = img1.reshape((h*w,c))

# grid search
# for i in range(3):
#     grid_search(3,img1,i)
# plt.show()
# sys.exit()

if is_newfile or not os.path.exists(new_save_folder):
    prepare_Matrix("img1",img1)
else:
    print("Pre-computed similarity matrix (W) and degree matrix (D) already exist!")
W1,D1,spacial,color = load_Matrix("img1")
W1,D1 = kernel2Matrix(spacial,color,[0.0001,1])
plt.figure()
plt.imshow(W1)
# label = k_means(img1,img1_data,4)
# label = kernel_k_means(img1,W1,4,True)
gif.append(img1)

start_time = time.time()

label = spectral(3,W1,D1,True,img1,is_show=False)

end_time = time.time()
time_c= end_time - start_time
min_c = int(time_c / 60)
time_c = time_c - min_c * 60
print('Clustering total time cost : {}m , {:.3f}s'.format(min_c,time_c))

plt.figure()
plt.imshow(label2Image(label,img1))
plt.figure()
plt.imshow(img1)
gif.save("Result/result.gif")
plt.show()