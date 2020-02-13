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

    def cartesianLabel2Polar(self, cartesianLabel, rotation=0):
        '''

        :param cartesianLabel: C*N*2, where C is the number of contour labels, N is number of points, 2 is 2 coordinates x and y;
        :param rotation: rotation angular in integer degree of (0,360)
        :return: polarLabel: (t,r) in size (C,N,2)

        '''

        x = cartesianLabel[:, :, 0] - self.centerx # cartesian x points to East, image x also points to East
        y = self.centery - cartesianLabel[:, :, 1]  # cartesian y points to North, but image y points to South
        r = np.sqrt(x ** 2 + y ** 2)
        t = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)  # size: C,N
        t = t * 360 / (2 * np.pi)

        # avoid 360 is expressed into 0 at the final coordinate
        t[:,self.tMax-1] = np.where(t[:,self.tMax-1] == 0, 359.9, t[:,self.tMax-1])

        rotation = rotation % 360
        if 0 != rotation:
            r = np.roll(r, rotation, axis=1)
        polarLabel = np.concatenate((np.expand_dims(t, axis=-1), np.expand_dims(r, axis=-1)), axis=-1)
        return polarLabel

    def polarLabel2Cartesian(self, polarLabel, rotation=0):
        '''

        :param polarLabel:
        :param rotation: the previous rotation from cartesian to polar
        :return: cartesianLabel: in C*N*2, where 2 is (x,y)
        '''
        t = polarLabel[:, :, 0]  # size:C*N
        r = polarLabel[:, :, 1]  # size:C*N
        rotation = rotation % 360
        if 0 != rotation:
            r = np.roll(r, -rotation, axis=1)
        t = t * 2 * np.pi / 360
        x = r * np.cos(t) + self.centerx  # cartesian x points to East, image x also points to East
        y = self.centery - r * np.sin(t)  # cartesian y points to North, but image y points to South
        cartesianLabel = np.concatenate((np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)), axis=-1)
        return cartesianLabel

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
        rotation = rotation % 360
        if 0 != rotation:
            polarImage = np.roll(polarImage, rotation, axis=1)
        polarLabel = self.cartesianLabel2Polar(cartesianLabel, rotation)
        return polarImage, polarLabel



    def polarImageLabel2Cartesian(self, polarImage, polarLabel, rotation=0):
        rotation = rotation % 360
        if 0 != rotation:
            polarImage = np.roll(polarImage, -rotation, axis=1)
        polarImage = cv2.rotate(polarImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cartesianmage = cv2.warpPolar(polarImage, self.cartesianImageShape, (self.centerx, self.centery), self.rMax,
                                   flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        cartesianLabel= self.polarLabel2Cartesian(polarLabel, rotation)
        return cartesianmage, cartesianLabel



