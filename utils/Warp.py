import numpy as np
import cv2

def dist(p1, p2):
    return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def affine(x, y):
    b = np.reshape(y, len(y)*2)
    A = np.zeros((len(y)*2, 6))
    for i in range(A.shape[0]):
        if i % 2 == 0:
            t = i//2
            A[i][0] = x[t][0]
            A[i][1] = x[t][1]
            A[i][2] = 1
        
        else:
            t = (i-1)//2
            A[i][3] = x[t][0]
            A[i][4] = x[t][1]
            A[i][5] = 1
    
    coef = np.linalg.inv(A.T @ A) @ A.T @ b
    M = np.array([[coef[0], coef[1], coef[2]],[coef[3], coef[4], coef[5]]], dtype=np.float64)
    return M

def Image_Warping(img1, img2, src, dst):
    # Warp Img2 on Img1 based on src and dst points
    M = np.zeros((3,3))
    M[2,2] = 1
    M[0:2][0:3] = affine(dst, src)
    T = np.linalg.inv(M)
        
    dst_img = np.zeros(img1.shape)
    Forward_locs = np.ones((img1.shape[0]*img1.shape[1], 3))
    Forward_locs[:, 0] = np.tile(np.arange(0, img1.shape[0]), img1.shape[1])
    Forward_locs[:, 1] = np.squeeze(np.reshape(np.tile(np.arange(0, img1.shape[1])[:,np.newaxis],
                                                       img1.shape[0]), (img1.shape[0]*img1.shape[1], 1)))
    
    backward_pts = (Forward_locs@T.T)[:,:2]
    backward_pts[:,0][backward_pts[:,0] > img2.shape[0]-1] = img2.shape[0]-1
    backward_pts[:,0][backward_pts[:,0] < 0] = 0        
    backward_pts[:,1][backward_pts[:,1] > img2.shape[1]-1] = img2.shape[1]-1
    backward_pts[:,1][backward_pts[:,1] < 0] = 0    
    logic1 = np.logical_and(backward_pts[:,0] > 0, backward_pts[:,0] < img2.shape[0]-1)
    logic2 = np.logical_and(backward_pts[:,1] > 0, backward_pts[:,1] < img2.shape[1]-1)
    logic = np.logical_and(logic1, logic2)
    inregion_pts = backward_pts[np.where(logic)[0],:]    
    # corner1 ........ corner3
    #   .                .
    #   .                .
    #   .                .
    # corner2 ........ corner4
    interpolated_pixels = np.zeros((backward_pts.shape[0], 3))
    interpolated_pixels[np.where(~logic)[0],:] = img2[np.int64(backward_pts[np.where(~logic)[0],:])[:,0],
                                                      np.int64(backward_pts[np.where(~logic)[0],:])[:,1],:].copy()
    temp = np.where(logic)[0]
    corner1 = np.int64(inregion_pts)
    corner2 = np.int64(inregion_pts) + np.array([1, 0])
    corner3 = np.int64(inregion_pts) + np.array([0, 1])
    corner4 = np.int64(inregion_pts) + 1
    area = (inregion_pts - corner1)[:,0] * (inregion_pts - corner1)[:,1]
    for i in range(3):
        interpolated_pixels[temp,i] += img2[corner4[:,0], corner4[:,1], i]*area
    
    area = (corner4 - inregion_pts)[:,0] * (corner4 - inregion_pts)[:,1]
    for i in range(3):
        interpolated_pixels[temp,i] += img2[corner1[:,0], corner1[:,1], i]*area
    
    area = abs((inregion_pts - corner2)[:,0] * (inregion_pts - corner2)[:,1])
    for i in range(3):
        interpolated_pixels[temp,i] += img2[corner3[:,0], corner3[:,1], i]*area
    
    area = abs((inregion_pts - corner3)[:,0] * (inregion_pts - corner3)[:,1])
    for i in range(3):
        interpolated_pixels[temp,i] += img2[corner2[:,0], corner2[:,1], i]*area
    interpolated_pixels[np.where(~logic)[0],:] = 0
    dst_img[np.int64(Forward_locs[:,0]),
            np.int64(Forward_locs[:,1]),:] = interpolated_pixels
    
    dst_img = np.uint8(dst_img)
    return dst_img