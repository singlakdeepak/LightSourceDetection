import numpy as np
import math
import cv2
from scipy.optimize import nnls
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
Code for Azimuthal angle calculation
'''
def omg_calc(azm1, azm2, diffuse_const):
    dotp = np.sin(azm1)*np.sin(azm2) + np.cos(azm1)*np.cos(azm2)
    dotp = diffuse_const*dotp
    if dotp <0:
        return 0
    return dotp

def give_azm_angle(x,y):
    if (x>0)and (y<0):
        angle = 2*np.pi - math.atan(abs(y/x))
    elif (x>0) and (y>=0):
        angle = math.atan(abs(y/x))
    elif (x<0) and (y>=0):
        angle = np.pi - math.atan(abs(y/x))
    elif (x<0) and (y<0):
        angle = np.pi + math.atan(abs(y/x))
    elif (x==0):
        if (y>0):
            angle = 0.5*np.pi
        else:
            angle = 1.5*np.pi
    return angle

def linear_reg(x,y):
#     from sklearn import linear_model
#     clf = linear_model.LinearRegression()
#     clf.fit(x,y)
#     return clf.coef_
    coef,_ = nnls(x,y)
    return coef


def contour_voting(lum, azimuth_cor, no_lights,diffuse_consts):
    # no_normals: The number of pixels in luminance set.
    no_normals = len(lum)
    print("Solving for %d lights and no of silhouette pixels are %d."%(no_lights,no_normals))
    eps = 1e-3
    no_cors = len(azimuth_cor)
    assert(no_normals==no_cors)
    sorted_lum_ind = np.argsort(lum)
    
    # Sorted the luminances in descending order.
    sorted_lum_des = np.take(lum, sorted_lum_ind)[::-1]
    sorted_azm_des = np.take(azimuth_cor, sorted_lum_ind)[::-1]
    sorted_diffuse_consts = np.take(diffuse_consts,
                                   sorted_lum_ind)[::-1]
    
    np.random.seed(5)
    # j represent the lights

    
    # Initializing the azimuthal coordinates for the lights
    # The first light has azimuth equal to the azm coodinate of 
    # the point with maximum luminance. 
    # Now continuously, the factor of 2*pi*i/N_lights is added. 
    maximum_azm = sorted_azm_des[0]
    lights_azimuth_cor = np.ones(no_lights)*maximum_azm
    for light in range(no_lights):
        lights_azimuth_cor[light] += 2*np.pi*light/no_lights
    weights_lights = np.zeros(no_lights, 
                              dtype = np.float32)
#     print(lights_azimuth_cor)
    
    # omega_mat: weight for the luminances, it has been initialized to zero.
    omega_mat = np.zeros(no_normals, 
                         dtype = np.float32)
    # Maps the function omega to all the normals 
    omg_func = lambda t1,t2,diffuse: omg_calc(t1,t2,diffuse)
    vfunc = np.vectorize(omg_func)
    
    consts = np.zeros((no_normals,no_lights)) 

    #############Initialize diffuse consts###############
    for i in range(no_normals):
        consts[i] = vfunc(sorted_azm_des[i],
                                 lights_azimuth_cor,1)
    vals_light_lums = np.random.uniform(np.min(lum),np.max(lum), no_lights)
    #####################################################
    
    
    prev_azimuth_cor = np.zeros_like(lights_azimuth_cor)
    itr = 0
    A=True
    while (np.linalg.norm(abs(lights_azimuth_cor- prev_azimuth_cor)) >eps)and A:
        # i represent the normals
        prev_azimuth_cor = np.copy(lights_azimuth_cor)
        for i in range(no_normals):
            
            # Total weight for the current luminance
            # light_azimuth_cor is an array of all the lights.
            # I have randomly filled the diffuse constant.
#             omega_mat[i] = np.sum(vfunc(sorted_azm_des[i], 
#                                          lights_azimuth_cor, 
#                                          sorted_diffuse_consts[i]))
            omega_mat[i] = np.sum(vfunc(sorted_azm_des[i], 
                                         lights_azimuth_cor, 
                                         vals_light_lums))
            for j in range(no_lights):
#                 alpha_i_j = sorted_lum_des[i]*omg_calc(sorted_azm_des[i],
#                                            lights_azimuth_cor[j],
#                                            sorted_diffuse_consts[i])
                alpha_i_j = sorted_lum_des[i]*omg_calc(sorted_azm_des[i],
                                           lights_azimuth_cor[j],
                                           vals_light_lums[j])
#                 print(alpha_i_j)
                if (omega_mat[i]!=0):
                    alpha_i_j /=omega_mat[i] # Weight of normal i
                else:
                    alpha_i_j = 0
                lights_azimuth_cor[j] = weights_lights[j]*lights_azimuth_cor[j] + \
                                         alpha_i_j*sorted_azm_des[i]
                weights_lights[j] += alpha_i_j
                if weights_lights[j]!=0 :
                    lights_azimuth_cor[j] /= weights_lights[j]
                else:
                    lights_azimuth_cor[j] = 0
            
            
            ##############################Not sure about this#############
        for i in range(no_normals):
            consts[i] = vfunc(sorted_azm_des[i],
                                     lights_azimuth_cor,
                                      1)
#         print(consts)
        print('Values of light lums are: ',vals_light_lums)
        vals_light_lums = linear_reg(consts,lum)
#         vals_light_lums = linear_reg(consts,sorted_lum_des)
            ##############################
        itr +=1
        print("Iteration %d completed. Moving next."%itr)
#         print(prev_azimuth_cor)
        print("The azimuthal angles at this iteration are: ",lights_azimuth_cor)
#         A = False
    
    
    print("Converged at the %d iteration."%itr)
    print("Light azimuthal angles are : ",lights_azimuth_cor*180/np.pi)
    return lights_azimuth_cor
    




'''
Code for Zenith Angle Calculation
'''
def createLineIterator(P1, P2, img):
    """
    ********Taken from Stack Overflow.************** 
    Produces and array that consists of the coordinates and 
    intensities of each pixel in a line between two points.
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1Y = P1[0]
    P1X = P1[1]
    P2Y = P2[0]
    P2X = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


def see_case(azm, divX, divY):
    value = np.sin(azm)*divX + np.cos(azm)*divY
    return value

def find_first_local(itbuffer, maxima = True):
    if maxima:
        local_maximas = argrelextrema(itbuffer[:,2], np.greater)
        if len(local_maximas[0])>0:
#             print(local_maximas[0])
            return (itbuffer[local_maximas[0][0],1],itbuffer[local_maximas[0][0],0])
        else: raise ValueError()
    else:
        local_minimas = argrelextrema(itbuffer[:,2], np.less)
        if len(local_minimas[0])>0: 
            return (itbuffer[local_minimas[0][0],1],itbuffer[local_minimas[0][0],0])
        else: raise ValueError()

def surface_normal_formula(P1,P2, ph_or_plo):
    P1Y, P1X = P1
    P2Y, P2X = P2
    ph_or_ploY, ph_or_ploX = ph_or_plo
    centreY,centreX = (P1Y + P2Y)/2 , (P1X + P2X)/2
    rad  = np.sqrt((centreY-P1Y)**2 + (centreX - P1X)**2)
#     print(rad)
    phdist = np.sqrt((centreY-ph_or_ploY)**2 + (centreX - ph_or_ploX)**2)
#     print(phdist)
    coshtheta = np.arccos(phdist/rad)
    if (ph_or_ploY<centreY):
        return (np.pi - coshtheta)
    else: return coshtheta


def calculate_zenith_angle(Image, Silhouette,
                            n_lights,azms, 
                          light_angles,
                          azmX, azmY):
    Image = cv2.bilateralFilter(Image,9,75,75)

    Rimg, Gimg, Bimg = Image[:,:,0], Image[:,:,1], Image[:,:,2]
    luminance = 0.2989 * Rimg + 0.5870 * Gimg + 0.1140 * Bimg
    print('Light angles are', light_angles)
#     print('Azimuthal angles of pixels are', azms)
    divImg = np.gradient(luminance)
    
    divImgY, divImgX = divImg[0], divImg[1]
    del divImg
    
    azm_into_div = lambda azm,divX, divY: see_case(azm,divX,divY)
    vfunc = np.vectorize(azm_into_div)    
    
    for i in range(n_lights):
        if (light_angles[i]<np.pi): 
            testangle_2 = light_angles[i] + np.pi
        else: testangle_2 = light_angles[i] - np.pi

        index_to_start = np.where(abs(light_angles[i] -azms)<1e-2)
        index_to_end = np.where(abs(testangle_2 - azms)<1e-2)
#         print(index_to_start)
#         print(index_to_end)
        azmX_to_start, azmY_to_start = azmX[index_to_start],azmY[index_to_start]
        azmX_to_end, azmY_to_end = azmX[index_to_end],azmY[index_to_end]
        len_start_inds = len(azmX_to_start)
        len_end_inds = len(azmX_to_end)
#         print(azmY_to_start,azmX_to_start)
#         print(azmY_to_end,azmX_to_end)
        for start in range(1):
            maxima = False
            P1 = (azmY_to_start[start],azmX_to_start[start])
            DirectionDerv = vfunc(light_angles[i],
                                     divImgX,
                                    divImgY)
            if DirectionDerv[P1]>0:
                maxima = True
            print(maxima)
            for end in range(1):
                P2 = (azmY_to_end[end],azmX_to_end[end])
                itbuffer = createLineIterator(P1,P2,luminance)
                if maxima:
                    pixel = find_first_local(itbuffer,maxima=maxima)
                    zenith_angle = surface_normal_formula(P1,P2,pixel)
                    print('Zenith angle obtained is: ', zenith_angle*180/np.pi)
                else:
                    Derivs_along_line = DirectionDerv[itbuffer[:,1].astype(np.uint),
                                                      itbuffer[:,0].astype(np.uint)]
                    zero_crossings = np.where(np.diff(np.sign(Derivs_along_line)))[0]
                    if (len(zero_crossings)!=0):
#                         print(zero_crossings[0])
                        pixel = (itbuffer[zero_crossings[0],1],
                                 itbuffer[zero_crossings[0],0])
                        zenith_angle = surface_normal_formula(P1,P2,pixel) - np.pi/2
#                         print(P1,P2,pixel)
                        print('Zenith angle obtained is: ', zenith_angle*180/np.pi)
        # Returned in the form (Y,X)
#                 print(find_first_local(itbuffer,maxima=DirectionDerv))

def give_images_light_detection(Image, Silhouette, n_lights = 3):
    shapeImg = Image.shape
    shapeSilhoutte = Silhouette.shape
#     assert(shapeImg == shapeSilhoutte)
    boundaryIndsY, boundaryIndsX = np.where(Silhouette!=0)
    Rimg, Gimg, Bimg = Image[:,:,0], Image[:,:,1], Image[:,:,2]

    # Finding the luminances of the silhouette pixels
    silhR, silhG, silhB = Rimg[boundaryIndsY,boundaryIndsX], \
                            Gimg[boundaryIndsY,boundaryIndsX], \
                            Bimg[boundaryIndsY,boundaryIndsX]
#     luminance = 0.2126*Rimg +0.7152*Gimg + 0.0722*Bimg
    luminance = 0.2989 * Rimg + 0.5870 * Gimg + 0.1140 * Bimg
    silhpixs = np.zeros_like(luminance)
    silhpixs[boundaryIndsY,boundaryIndsX] = vectorlum = luminance[boundaryIndsY,boundaryIndsX]
    
    # The origin taken as center of the photograph
    ymed,xmed = shapeSilhoutte
    xmed= xmed/2
    ymed = ymed/2
    print("The center is ",(xmed,ymed))
    # Finding azimuthal angles of the boundaries
    azmX, azmY = boundaryIndsX - xmed, boundaryIndsY - ymed
    find_angles = np.vectorize(give_azm_angle)
    azms = find_angles(azmX,azmY)
#     print(azms)
    print("no of Silhouette pixels are: ",len(azms))
    diffuse_consts = np.random.uniform(3,4, len(vectorlum)) # How to choose them???
    light_azimuth_angles =  contour_voting(vectorlum,azms,n_lights,diffuse_consts)
    
#     light_azimuth_angles = np.array([167.709296 ,134.57994889])*np.pi/180
#     light_azimuth_angles = np.array([201.24,279.64])*np.pi/180

    calculate_zenith_angle(Image, Silhouette, n_lights,
                           azms, light_azimuth_angles,
                          boundaryIndsX,boundaryIndsY) 
#     print(boundaryIndsY,boundaryIndsX)
#     print(azms*180/np.pi)
