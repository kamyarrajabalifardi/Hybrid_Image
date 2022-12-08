import numpy as np
import cv2
def Gaussian_Filter(img, sigma, lh_type = 'low', direction = 'both'):
    # Designing 2D gaussian filter with defined signma
    
    n = np.arange(-img.shape[0]//2, img.shape[0]//2+1, 1)
    m = np.arange(-img.shape[1]//2, img.shape[1]//2+1, 1)
    G_X = np.exp(-pow(n,2)/(2*pow(sigma, 2)))
    G_Y = np.exp(-pow(m,2)/(2*pow(sigma, 2)))
    
        
    # G_X = G_X/sum(G_X)
    # G_Y = G_Y/sum(G_Y)    
    G_Y.shape = (len(m), 1)
    G_X.shape = (1,len(n))

    if direction == 'both':
        if lh_type=='low':
            return (G_Y@G_X).T
        if lh_type=='high':
            temp = G_Y@G_X
            temp = temp/np.max(temp)
            return 1 - (temp).T

    if direction == 'ver':
        if lh_type=='low':
            return G_Y
        if lh_type=='high':
            return 1-G_Y
    
    if direction == 'hor':
        if lh_type=='low':
            return G_X
        if lh_type=='high':
            return 1 - G_X
        
def Normalized_Img(img, Truncating = False):
    # This function works for both grayscale and RGB images
    # Arguments:
        #   img --- input image
        #   Truncating --- Truncating values more than 255 and less than 0 or not
    # Output:
        #   Image scaled between 0 and 255
        
    Image = img.copy()
    Image = np.float64(Image)
    
    if Truncating == True:
        Image[Image < 0] = 0
        Image[Image > 255] = 255
        
    try:
        for i in range(3):    
            Image[:,:,i] = Image[:,:,i] - np.min(Image[:,:,i])
            Image[:,:,i] = Image[:,:,i]/np.max(Image[:,:,i])*255
            
    except:
        Image = Image - np.min(Image)
        Image = Image/np.max(Image)*255  
    return Image    