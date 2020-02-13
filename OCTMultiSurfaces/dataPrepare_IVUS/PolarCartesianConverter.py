import cv2
import numpy as np

class PolarCartesianConverter():
    def __init__(self, cartesianImageShape, centerx,centery, rMax, tMax=360):
        '''

        :param cartesianImageShape: (H,W)
        :param centerx: Cartesian center x
        :param centery: Cartesian center y
        :param rMax: radial Max
        :param tMax: angular Max in [0,360] scale.
        '''
        self.cartesianImageShape = cartesianImageShape
        self.centerx = centerx
        self.centery = centery
        self.rMax = rMax
        self.tMax = tMax

    def cartesianImageLabel2Polar(self, cartesianImage, cartesianLabel, rotation=0):
        '''

        :param cartesianImage: H*W array
        :param cartesianLabel: C*N*2, where C is the number of class labels, N is number of points, 2 is 2 coordinates x and y;
        :param rotation: rotation angular in integer degree of (0,360)
        :return: polarImage, in size(self.rMax, 360)
                 polarLabel: (t,r) in size (C,N,2)

        '''

        polarImageSize = (self.rMax, self.tMax)
        polarImage = cv2.warpPolar(cartesianImage,polarImageSize, (self.centerx,self.centery), self.rMax, flags=cv2.WARP_FILL_OUTLIERS)
        polarImage = cv2.rotate(polarImage, cv2.ROTATE_90_CLOCKWISE)
        x = cartesianLabel[:, :, 0] - self.centerx
        y = cartesianLabel[:, :, 1] - self.centery
        r = np.flip(np.sqrt(x**2 + y**2),axis=1) # size: C,N, flip because warpPolar is clockwise direction to warp.
        t = (np.arctan2(y,x)+2*np.pi)%(2*np.pi)  # size: C,N
        t = t*360/(2*np.pi)

        rotation = rotation%360
        if 0 != rotation:
            polarImage = np.roll(polarImage, rotation, axis=1)
            r = np.roll(r, -rotation, axis=1)
        polarLabel = np.concatenate((np.expand_dims(t, axis=-1),np.expand_dims(r,axis=-1)), axis=-1)
        return polarImage, polarLabel

    def polarImageLabel2Cartesian(self, polarImage, polarLabel, rotation=0):
        r = polarLabel[:, :, 1] # size:C*N
        t = polarLabel[:, :, 0] # size:C*N
        rotation = rotation % 360
        if 0 != rotation:
            polarImage = np.roll(polarImage, -rotation, axis=1)
            r = np.roll(r, rotation, axis=1)
        t = t*2*np.pi/360
        r = np.flip(r, axis=1)
        x = r*np.cos(t) + self.centerx
        y = r*np.sin(t) + self.centery
        polarImage = cv2.rotate(polarImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cartesianmage = cv2.warpPolar(polarImage, self.cartesianImageShape, (self.centerx, self.centery), self.rMax,
                                   flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        cartesianLabel= np.concatenate((np.expand_dims(x,axis=-1), np.expand_dims(y,axis=-1)), axis=-1)
        return cartesianmage, cartesianLabel



