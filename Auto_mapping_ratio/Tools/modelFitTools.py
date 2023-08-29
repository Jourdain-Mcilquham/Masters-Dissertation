# -*- coding: utf-8 -*-

import numpy as np
import os
from scipy import ndimage

'import spectral Python'
import spectral.io.envi as envi

'Import user defined libraries'
from generalUtilities import generalUtilities
from hsiUtilities import hsiUtilities

"""
Module taken from Saranath et al 2021 github repository

https://github.com/arunsaranath
"""



class modelFitTools():
    def __init__(self, strtBand=0, numBands=240, imgType='FRT'):
        self.strtBand = strtBand
        self.numBands = numBands
        self.stopBand = self.strtBand + self.numBands

        if(imgType == 'FRT') or (imgType == 'FRS') or (imgType == 'ATO'):
            self.strtRow = 1
            self.stopRow = -2
            self.strtCol = 29
            self.stopCol = -31
        else:
            if (imgType == 'HRL') or (imgType == 'HRS'):
                self.strtRow = 0
                self.stopRow = 0
                self.strtCol = 15
                self.stopCol = -4
            else:
                if (imgType == 'Unk'):
                    self.strtRow = 0
                    self.stopRow = 0
                    self.strtCol = 0
                    self.stopCol = 0

                else:
                    raise ValueError('Unknown CRISM data Type')



    def find_non_zero_bounds(self, img):


            # Rows
            start_row = 0
            end_row = img.shape[0] - 1  # Default to last row

            # Find the first row without zeros
            for i in range(img.shape[0]):
                if np.count_nonzero(img[i, :]) == img.shape[1]:  # If all elements are non-zero
                    start_row = i
                    break

            # Find the last row without zeros
            for j in range(img.shape[0]-1, -1, -1):
                if np.count_nonzero(img[j, :]) == img.shape[1]:  # If all elements are non-zero
                    end_row = j
                    break

            # Columns
            start_col = 0
            end_col = img.shape[1] - 1  # Default to last column

            # Find the first column without zeros
            for i in range(img.shape[1]):
                if np.count_nonzero(img[:, i]) == img.shape[0]:  # If all elements are non-zero
                    start_col = i
                    break

            # Find the last column without zeros
            for j in range(img.shape[1]-1, -1, -1):  # Start from the last column and move leftwards
                if np.count_nonzero(img[:, j]) == img.shape[0]:  # If all elements are non-zero
                    end_col = j
                    break

           
            self.strtRow  = start_row
            self.stopRow  = end_row - img.shape[0]
            self.strtCol  =  start_col
            self.stopCol  = end_col - img.shape[1]
        


    def crismImgSmooth(self, imgName, filterSize=5):
        """
        This function smooths each band of the hyperspectral image band by band

        :param imgName: Name of the image to be smoothed
        :param filterSize: The kernel size of the boxcar(uniform) filter
        :return:
        """
        imgHdrName = imgName.replace('.img', '.hdr')
        header = envi.read_envi_header(imgHdrName)

        'Read in the background image'
        crImg = envi.open(imgHdrName)
        crCube = crImg.load().copy()
        self.find_non_zero_bounds(crCube)
        [rows, cols, bands] = crCube.shape

        'Initialize matrix to nans'
        crCube_smoothed = np.empty((rows, cols, bands), dtype=float)
        crCube_smoothed[:] = np.nan

        for ii in range(self.strtBand, self.stopBand, 1):
            bandImg = np.squeeze(crCube[self.strtRow:(rows + self.stopRow), self.strtCol:(cols + self.stopCol), ii])
            bandImg_smooth = ndimage.uniform_filter(bandImg, size=filterSize)
            crCube_smoothed[self.strtRow:(rows + self.stopRow),
            self.strtCol:(cols + self.stopCol), ii] = bandImg_smooth

        outFileName = f'smoothed_{filterSize}.hdr'
        envi.save_image(outFileName, crCube_smoothed, dtype=np.float32, force=True,
                        interleave='bil', metadata=header)

        return outFileName


    '------------------------------------------------------------------------------------------------------------------'
    'If a image has missing columns fill each row based on its neighbors'
    '------------------------------------------------------------------------------------------------------------------'
    def crism_fillNanRows(self, imgName):
        """
        This function fill the empty of NaN rows using the neighbors.

        :param imgName: Name/address of the image to be smoothed
        :return:
        """

        hdrName = imgName.replace('.img', '.hdr')
        header = envi.read_envi_header(hdrName)
        'Read in the background image'
        crImg = envi.open(hdrName)
        crCube = crImg.load().copy()
        [rows, cols, bands] = crCube.shape

        arrCrImg = crCube.reshape((rows * cols, bands))
        arrCrImgCrop = arrCrImg[:, self.strtBand:self.stopBand]
        'Fill the NaNs in the columns'
        arrCrImgCrop = generalUtilities().fill_nan(arrCrImgCrop)
        'Fill the NaNs in the rows'
        arrCrImgCrop = generalUtilities().fill_nan(arrCrImgCrop.T)
        arrCrImg[:, self.strtBand:self.stopBand] = arrCrImgCrop.T
        'Reshape to image size'
        crCube_nr = arrCrImg.reshape((rows, cols, bands))

        # 'Save the background image'
        # outFileName1 = imgName.replace('_CR', '_CRnR')
        outFileName1 = f'{imgName[:-4]}_CRnR.hdr'

        envi.save_image(outFileName1, crCube_nr, dtype='single',
                        force=True, interleave='bil', metadata=header)

        return outFileName1