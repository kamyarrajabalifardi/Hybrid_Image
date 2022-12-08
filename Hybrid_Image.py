import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.Filters import *
from utils.Warp import *

global low_img_fft
global high_img_fft


def nothing(x):
    # Do nothing!!
    pass

def Hybrid_Image(LP_sigma, HP_sigma, alpha, beta):
    # Create a hybrid image based on two images in fourier domain
    # input:
    # --LP_Sigma: sigma of low pass filter 
    # --HP_Sigma: sigma for high pass filter
    # --alpha, beta: coefficients of linear combination of lowpass and highpass images
    # output:
    # hybbrid_img
    G_LP = Gaussian_Filter(low_img,  LP_sigma, lh_type = 'low')[:-1,:-1]
    G_HP = Gaussian_Filter(high_img, HP_sigma, lh_type = 'high')[:-1,:-1]    


    hybrid_img_fft = alpha * np.multiply(low_img_fft,  G_LP[:,:, np.newaxis]) +\
                     beta  * np.multiply(high_img_fft, G_HP[:,:, np.newaxis])
                        
    hybrid_img = np.zeros(hybrid_img_fft.shape)
    for i in range(hybrid_img_fft.shape[2]):
        hybrid_img[:,:, i] = np.real(np.fft.ifft2(np.fft.ifftshift(hybrid_img_fft[:, :, i])))

    hybrid_img_near = np.uint8(Normalized_Img(hybrid_img))
    hybrid_img_far = cv2.resize(hybrid_img_near, (hybrid_img_near.shape[1]//5,
                                                  hybrid_img_near.shape[0]//5))
    hybrid_img = np.zeros((hybrid_img_near.shape[0], hybrid_img_near.shape[1] + hybrid_img_far.shape[1], 3))
    hybrid_img[:, :hybrid_img_near.shape[1], :] = hybrid_img_near
    hybrid_img[:hybrid_img_far.shape[0], hybrid_img_near.shape[1]:, :] = hybrid_img_far    
    return np.uint8(hybrid_img)


# Functions used for finding corresponding points between two images with the
# help of user. (Corresponding points should be chosen in order!)
def Click_Point1(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
        img_copy1[y-5:y+5,x-5:x+5,:] = (0,255,0)
        cv2.imshow(title1, img_copy1)
        X1.append(y)
        Y1.append(x)
def Click_Point2(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
        img_copy2[y-5:y+5,x-5:x+5,:] = (0,255,0)
        cv2.imshow(title2, img_copy2)
        X2.append(y)
        Y2.append(x)
        
def Choosing_Points1(img1, img2, scale1=3, scale2=5):
    global X1, X2
    global Y1, Y2
    global img_copy1, img_copy2
    global title1, title2
    title1 = 'img1'
    title2 = 'img2'
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []

    img_copy1 = cv2.resize(img1.copy(), (img1.shape[1]//scale1, img1.shape[0]//scale1))
    img_copy2 = cv2.resize(img2.copy(), (img2.shape[1]//scale2, img2.shape[0]//scale2))
    cv2.imshow(title1, img_copy1)
    cv2.imshow(title2, img_copy2)
    cv2.setMouseCallback(title1, Click_Point1)
    cv2.setMouseCallback(title2, Click_Point2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (np.array(X1)*scale1, np.array(Y1)*scale1,
            np.array(X2)*scale2, np.array(Y2)*scale2)




if __name__ == '__main__':
    img1 = cv2.imread("Img1.jpg")
    img2 = cv2.imread("Img2.jpg")
    x1,y1,x2,y2 = Choosing_Points1(img1, img2)
    
    src = np.float32(np.concatenate((x1[:, np.newaxis], y1[:,np.newaxis]), axis = 1))
    dst = np.float32(np.concatenate((x2[:, np.newaxis], y2[:,np.newaxis]), axis = 1))
    dst_img = Image_Warping(img1, img2, src, dst)
    
    
    low_img = img1
    high_img = dst_img
    if low_img.shape[0] % 2 == 0:
        low_img = low_img[0:-1, :, :]
        high_img = high_img[0:-1, :, :]
    if low_img.shape[1] % 2 == 0:
        low_img = low_img[:, 0:-1, :]
        high_img = high_img[:, 0:-1, :]
    low_img_fft = np.zeros(low_img.shape, dtype=np.complex_)
    high_img_fft = np.zeros(high_img.shape, dtype=np.complex_)
    
    for i in range(low_img_fft.shape[2]):
        low_img_fft[:,:,i] = np.fft.fftshift(np.fft.fft2(low_img[:,:,i]))
    
    for i in range(high_img_fft.shape[2]):
        high_img_fft[:,:,i] = np.fft.fftshift(np.fft.fft2(high_img[:,:,i]))
    
    
    title = "Parameters"
    LP_Sigma_MAX = 60
    HP_Sigma_MAX = 60
    
    # Creating Trackbars of window     
    cv2.namedWindow(title)
    cv2.createTrackbar("LP_Sigma: ", title, 1, LP_Sigma_MAX, nothing)
    cv2.createTrackbar("HP_Sigma: ", title, 1, HP_Sigma_MAX, nothing)
    cv2.createTrackbar("Alpha: ", title, 1, 10, nothing)
    cv2.createTrackbar("Beta: ", title,  1, 10, nothing)
    cv2.createTrackbar("Save",   title, 0, 1, nothing)
    cv2.resizeWindow(title, 500, 30)
    
    hybrid_img = high_img
    
    while True:
        cv2.imshow('Hybrid_Img', cv2.resize(hybrid_img,
                                           (hybrid_img.shape[1]//3, hybrid_img.shape[0]//3)))
        
        
        # Esc button to end the program
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
        # retreiving the values of trackbars
        LP_Sigma = cv2.getTrackbarPos("LP_Sigma: ", title)
        HP_Sigma = cv2.getTrackbarPos("HP_Sigma: ", title)
        alpha = cv2.getTrackbarPos("Alpha: ", title)
        beta = cv2.getTrackbarPos("Beta: ", title)
        Save = cv2.getTrackbarPos("Save", title)
        hybrid_img = Hybrid_Image(LP_Sigma, HP_Sigma, alpha, beta)
        if Save == 1:
            cv2.imwrite('Result.jpg', hybrid_img)
    cv2.destroyAllWindows()    
    
    
