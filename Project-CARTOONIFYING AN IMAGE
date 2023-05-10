import cv2

import numpy as np

def read_img(filename):        #for inputting image
 img= cv2.imread(filename)     #read image and return image matrix
 return img


def edge_detection(img,line_wdt,blur):   
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make grayscale image- b&w image
    grayBlur = cv2.medianBlur(gray,blur)    #smooth image and help in edge detection- gives blurred b&w image
    edges= cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_wdt,blur) #will find the edges
    return edges    #give sketch image


def color_quantisation(img,k):  #k-no. of clusters
    data= np.float32(img).reshape((-1,3)) #rows will be increasing but column will be 3
    criteria= (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20,0.001) #20-maxiter to find centroid, 0.001-accuracy
    ret, label, center= cv2.kmeans(data,k,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #centroid 10- no. of attempts algorithm is executed
    center= np.uint8(center)   #change into integer value
    result= center[label.flatten()] #we will have label that will consist value that is position of centres(of groups of colors)
    result= result.reshape(img.shape)
    return result   #give painiting image


img = read_img('C:\\Users\\Deepika\\Desktop\\project\\imgg2.jpg') #it will have image matrix
line_wdt= 9 #can change to any odd number
blur_value=5
totalColors=8 #as less as possible 4or greater than 4

edgeImg= edge_detection(img, line_wdt, blur_value) #pass the values

img= color_quantisation(img,totalColors) #k will have total clusters that is no of group of colors

blurred= cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200) #remove noise in group of 7

cartoon= cv2.bitwise_and(blurred, blurred, mask=edgeImg)   #merge both image- overlap each other

cv2.imwrite('cartoon.jpg', cartoon)
cv2.imwrite('edge.jpg', edgeImg)
cv2.imwrite('painting.jpg', img)

