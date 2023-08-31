import pandas as pd
import numpy as np
import spectral
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
import h5py
import spectral
from scipy import interpolate
import spectral.io.envi as envi
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.filters import uniform_filter1d as filter1d


class get_CRISM():
    def __init__(self, base_path, obj_path):
        self.base_path = base_path 
        self.obj_path = obj_path

        img = envi.open(r'Corrected_images/FRT0000A425_cr.hdr',r'Corrected_images/FRT0000A425_cr.img')
        header = envi.read_envi_header( r'Corrected_images/FRT0000A425_cr.hdr')
        wvl = header['wavelength']
        wvl = [float(element) for element in wvl]
        self.wvl = wvl[:240]
        self.counter = 0

    ''' Taken from Arun 2019 github'''
    def convex_hull_jit(self, wvl, spectrum):
        """Computes the convex hull of a set of 2D points.

        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
        starting from the vertex with the lexicographically smallest coordinates.
        Implements the algorithm CONVEXHULL(P) described by  Mark de Berg, Otfried
        Cheong, Marc van Kreveld, and Mark Overmars, in Computational Geometry:
        Algorithm and Applications, pp. 6-7 in Chapter 1

        :param points: A N X 2 matrix with the wavelengths as the first column
        :return: The convex hull vector
        """
        'The starting points be the first two points'
        xcnt, y = wvl[:2], spectrum[:2]
        'Now iterate over the other points'
        for ii in range(2, spectrum.shape[0], 1):
            'check next prospective convex hull members'
            xcnt = np.append(xcnt, wvl[ii])
            y = np.append(y, spectrum[ii])
            flag = True

            while (flag == True):
                'Check if a left turn occurs at the central member'
                a1 = (y[-2] - y[-3]) / (xcnt[-2] - xcnt[-3])
                a2 = (y[-1] - y[-2]) / (xcnt[-1] - xcnt[-2])
                if (a2 > a1):
                    xcnt[-2] = xcnt[-1]
                    xcnt = xcnt[:-1]
                    y[-2] = y[-1]
                    y = y[:-1]
                    flag = (xcnt.shape[0] > 2);
                else:
                    flag = False

        return np.vstack((xcnt, y))

    def fill_nan(self, data):

        """
        interpolate to fill nan values based on other entries in the row
        :param data: a numpy matrix with nan values
        :return: the matrix with the nan interpolated by the nan values
        """
        ok = ~np.isnan(data)
        xp = ok.ravel().nonzero()[0]
        fp = data[~np.isnan(data)]
        x = np.isnan(data).ravel().nonzero()[0]
        data[np.isnan(data)] = np.interp(x, xp, fp)
        return data


    def calculate_ratio_array(self, img, n, d, pix_size = 3, verbose = False):
        '''
        Calculates the ratioed spectrum of an image at specified mineral and 
        spectrally neutral zone locations.

            Args:
                img (str): Path to the image location, atmosphere corrected.
                n (list): Location of the mineral from the Viviano Beck paper (x, y).
                d (list): Location of the spectrally neutral zone (x, y).
                pix_size (int): Size of the neighborhood around which the pixel average
                                is created.

            Returns:
                tuple: A tuple containing:
                    img_array (ndarray): Numpy array of the ratioed spectrum with shape (M, 240),
                                        where M is the number of valid pixels after removing NaNs.
                    test_image (ndarray): Test image showing the mineral location for checking,
                                        with shape (240,)
        '''
        
    
        neighborhood_size = (pix_size, pix_size, 1)  

        # Open ENVI file
        img = spectral.open_image(img)
        
        # Read data as a 3D NumPy array
        data = img.load()

        # Convert data to a 3D NumPy array
        data_array = np.array(data)
        
        data_array = self.fill_nan(data_array)

        averaged_array = uniform_filter(data_array.copy(), size=neighborhood_size, mode='constant')
        # create the denominator
        denominator = averaged_array[int(d[1]),int(d[0])]

        if (verbose):
            averaged_array = uniform_filter(data_array.copy(), size=neighborhood_size, mode='constant')
            # create the denominator/numerator
            numerator = averaged_array[int(n[1]),int(n[0])]
            denominator = averaged_array[int(d[1]),int(d[0])]
            # remove the no data values
            numerator[numerator == 65535] = None
            denominator[denominator == 65535] = None
            test_image = numerator/denominator
            
        final_list = []
        
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                data_array[i,j][data_array[i,j] == 65535] = np.nan
                
                data_array[i,j] = data_array[i,j]/denominator
            
                f = data_array[i,j][4:244]
                
            
                 # Check if all elements in the list are nan and if so, skip
                if not(np.isnan(f).all()):
                    final_list.append(f)
        

        final_list = np.asarray(final_list)
        print(final_list.shape)
        return final_list

    def eliminate_unremarkable_spectra(self, data, threshold):
      

        smallest_value = np.nan_to_num(np.min(data))
        if(smallest_value <= 1 - threshold):
            return True
        else:
            return False

    def remove_cr(self, list, threshold = 0.02):
       
        cr_results = []
        for i in range(len(list)):
            list[i] = filter1d(list[i],11)
            cHull = self.convex_hull_jit(self.wvl,list[i])
            
            f = interpolate.interp1d(np.squeeze(cHull[0, :]), np.squeeze(cHull[1, :]))
            yc = f(self.wvl)
            cnt_rem = list[i]/yc
            
            if (self.eliminate_unremarkable_spectra(cnt_rem, threshold) and np.all(cnt_rem >= 0) and np.all(cnt_rem <= 1)):
                cr_results.append(cnt_rem)
            else:  
                self.counter += 1
            if(i%25000 == 0):   
                print(i)
            
        print(f'Number of unremarkable spectra: {self.counter}')
        print(len(cr_results))
        return cr_results
    
    
    def create_training_dataset(self):
        '''
        Input:
            img_basepath(str): Path to the corrected images directory
            object_path(str): Path to the object_ids file
            n - number of images to be turned into training data
        Output:
            training_data - Numpy array (240,), where N = number of spectra
        '''
        
        object_ids = pd.read_csv(self.obj_path)

        object_ids['N'] = [x.split() for x in object_ids['N']]
        object_ids['D'] = [x.split() for x in object_ids['D']]

        result_array = []
       

        for index, row in object_ids.iterrows():
            try:
                object_id = row[1]
                img_path = f"{self.base_path}/{object_id}_cr.hdr"
                
                # Open ENVI file
                print(f'{object_id} successfully loaded')
                
                result = self.calculate_ratio_array(img_path, row[2], row[3])
                
                result_array.extend(result)

            except Exception as e:
                print(f"An error occurred: {e}")
                print(f'{object_id} is not found or unreadable')

        print(f"length of result array = {len(result_array)}")
        
        
        cont_removed = self.remove_cr(result_array)
        
            

        print('Samples have been processed')
        data_array = np.array(cont_removed)
        
        return data_array
    
    def write_to_h(self, ls, location):
        df = pd.DataFrame(ls)
        store = pd.HDFStore(f'{location}.h5')
        store['data'] = df
        store.close()
   

base_path = r"Corrected_images"
obj_path = r"Object_ids3.csv"

obj = get_CRISM(base_path=base_path, obj_path=obj_path)
l = obj.create_training_dataset()
# data = np.array([[1, 2], [3, 4], [5, 6]])
# print(data.shape)
obj.write_to_h(l, 'Crism_test')


