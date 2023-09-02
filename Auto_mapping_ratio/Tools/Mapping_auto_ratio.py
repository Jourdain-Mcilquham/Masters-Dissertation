from minMapTools import minMapTools
from modelFitTools import modelFitTools
from crismProcessing_parallel  import crismProcessing_parallel
from hsiUtilities import hsiUtilities
from generalUtilities import generalUtilities
from models import Discriminator
import torch
import torch.nn as nn'
from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


"""
The is the main script for the mapping of the ratio images. It will load the pre-trained GAN model and the pre-trained
discriminator model. It will then load the spectra from the exemplars and scale them if needed. It will then reshape the
spectra to the shape of the input size of the GAN model. It will then return the scaled spectra and the discriminator
representation of the spectra. 

This code has been modified from Arun M Saranathan Github:

https://github.com/arunsaranath

"""





import os
from False_composite import False_composite

class Mapping_auto_ratio:
    def __init__(self):
        pass

    def GAN_REP(self):
        '''
        This function will load the pre-trained GAN model and the pre-trained discriminator model.
        It will then load the spectra from the SSA and scale them if needed. It will then reshape the
        spectra to the shape of the input size of the GAN model. It will then return the scaled spectra
        and the discriminator representation of the spectra.
            :param sLvl: The scale level to scale the spectra to.
            :param scaleFlag: Flag to scale the spectra or not.
            :return: The scaled spectra and the discriminator representation of the spectra.
        '''
        discriminator = Discriminator()
        discriminator_weights_path = 'wgan_gp_netD.pkl'
        discriminator = torch.load(discriminator_weights_path)
        # Remove the last layer
        dis_cut = nn.Sequential(*list(discriminator.children())[:-2])
        dis_cut = dis_cut.float()

        exemplars = pd.read_csv('Tools/exemplar_spectra.csv', header=None)
        exemplars = np.array(exemplars)

        for i in range(len(exemplars)):
            exemplars[i] = (exemplars[i] - np.min(exemplars[i])) / (np.max(exemplars[i]) - np.min(exemplars[i]))

        envi.save_image('Tools/exemplars_normalised.hdr', exemplars, dtype='float32', force=True, interleave='bil')
        return  exemplars, dis_cut
    
 
    def normalise_img(self, f_name):
        img = envi.open(f_name)
        img_data = img.load().copy()

        # Normalise img
        min_vals = np.min(img_data, axis=2, keepdims=True)
        max_vals = np.max(img_data, axis=2, keepdims=True)
        normalized_img = (img_data - min_vals) / (max_vals - min_vals + 1e-10)  # Add a small constant to avoid division by zero

        # Apply Savitzky-Golay filter
        for band in range(normalized_img.shape[2]):
            normalized_img[:, :, band] = savgol_filter(normalized_img[:, :, band], 9, 2)

        img_normalised = normalized_img[:, :, 4:244]
        print(img_normalised.shape)
        envi.save_image('normalised.hdr', img_normalised, dtype='float32', force=True, interleave='bil')
        return 'normalised.hdr'


   

    def create_map(self, f_name):
        endMem_Name = ''
        
        base_path = '../Ratio_hdrs/'

        # check if f_name directory exists if not create it
        dir_name = f_name[:-4]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        normalise_flag = True

        f_name = base_path + f_name

        exemplars, discriminator = self.GAN_REP()
        os.chdir(dir_name)

        # Normalise the image if needed
        f_name = self.normalise_img(f_name) if normalise_flag else f_name

        # Below loop taken from Saranath et al 2021 
        # https://github.com/arunsaranath

   
        'Since we are mapping at different mapping levels'
        for kernelSize in range(1,6,2):
            '--------------------------------------------------------------------------------------------------'
            'STEP-6: SMOOTHING CONTINUUM REMOVED IMAGE'
            '--------------------------------------------------------------------------------------------------'
            crSmoothImageName = modelFitTools(0, 240).crismImgSmooth(f_name, kernelSize)

            '--------------------------------------------------------------------------------------------------'
            'STEP-7: GENERATE SIMILARITY MAPS BETWEEN MICA REPRESENATIONS AND DATA REPRESENTATIONS'
            '--------------------------------------------------------------------------------------------------'
            print(f'Guessing {kernelSize}x{kernelSize} Minerals')
            simScoreMapName = minMapTools(0, 240).create_Maps4CRISMImages_Cosine(discriminator, exemplars,
                                                                                    crSmoothImageName, endMem_Name)

            '--------------------------------------------------------------------------------------------------'
            'STEP-8: IDENTIFY SIGNIFICANT PIXEL SPECTRA & THRESHOLD SIMILARITY MAPS TO MINIMIZE FALSE POSITIVES'
            '--------------------------------------------------------------------------------------------------'
            'First create a mask file that will hold the pixels of interest'
            maskName = minMapTools(0, 240).create_Mask4CRISMImages(crSmoothImageName)
            'Now create the best guess map'
            bestGuessName = minMapTools(4, 240).create_Maps4CRISMImages_BestGuess(simScoreMapName, maskName)

        compMapName = minMapTools(0, 240).create_Maps4CRISMImages_CompBestGuess(bestGuessName)
            
        return compMapName


    def create_color_map(self, compMapName, top_10 = False, selected_minerals = [], thresholds = {}, colour_matrix = {}):
        '''
            Function to create the colour map for the minerals
            :param compMapName: The name of the best guess map
            :param top_10: Boolean to determine if the top 10 minerals should be used
            :param selected_minerals: A list of the selected minerals
            :param thresholds: A dictionary of the thresholds for the minerals
            :param colour_matrix: A dictionary of the colours for the minerals
            :return: The guess map and the minerals used
        '''

        colMat = np.asarray(list(colour_matrix.values()), dtype=np.float32)
        thresholds = np.asarray(list(thresholds.values()), dtype=np.float32)
        # colMat = [x/255 for x in colMat]
        if top_10:
            top_10 = self.return_top_10_minerals(compMapName)
            # Generate the Guess map
            guessMap, minerals = minMapTools(0, 240).create_Maps4CRISMImages_GuessMap(compMapName, colMat, thresholds=thresholds, top_10=top_10)
        elif not top_10 and not selected_minerals:
            guessMap, minerals = minMapTools(0, 240).create_Maps4CRISMImages_GuessMap(compMapName, colMat, thresholds=thresholds)
        elif selected_minerals:
            guessMap, minerals = minMapTools(0, 240).create_Maps4CRISMImages_GuessMap(compMapName, colMat, thresholds=thresholds, selected_mineral=selected_minerals)


        return guessMap, minerals

    def produce_figure(self, guessmap, mineral_used, basemap, colour_matrix = {}, posx = 0.78, posy = 0.1, alpha = 1):
        """
            Function to produce the figure with the guess map and the mineral colours
            :param guessmap: The name of the guess map
            :param mineral_used: The minerals used
            :param basemap: The name of the base map
            :param colour_matrix: A dictionary of the colours for the minerals
            :param posx: The x position of the legend
            :param posy: The y position of the legend
            :param alpha: The alpha value of the guess map
            :return: The guess map
        """
        

        'Now plot the guess map'
        mineral_map = envi.open(guessmap)
        mineral_map = mineral_map.load()

        # THis should be computed on the size of the image
        ylim = [0, 480]
        xlim = [27, 600]

        rgb_mineral_map = np.dstack((mineral_map[:,:,0], mineral_map[:,:,1], mineral_map[:,:,2]))

        rgb_mineral_map = rgb_mineral_map.astype(float)/255.0

        transparent_mask = np.all(rgb_mineral_map == [0, 0, 0], axis=-1)
        alpha_channel = np.where(transparent_mask, 0, 1)

        rgba_image = np.dstack((rgb_mineral_map, alpha_channel))


        mat = False_composite().load_image(basemap)
        if_, rem = False_composite().filter_bad_pixels(mat['IF'])
        im_shape = False_composite().image_shape(mat)
        f = False_composite().get_false_colors(if_.reshape(*im_shape, -1), rem.reshape(im_shape))
        f = np.flip(f, axis = (0,1))

        
        # Convert dictionary keys to a list
        keys_list = list(colour_matrix.keys())

        # Create a new dictionary with only the selected keys
        selected_minerals = {keys_list[i]: colour_matrix[keys_list[i]] for i in mineral_used if i < len(keys_list)}

        # Find the number of lines needed to display the minerals
        num_lines = len(selected_minerals)


        patch_posx = posx
        patch_posy = posy
        patch_width = 0.25
        patch_height = (num_lines*0.045) + 0.05
        offset_x = 0.02
        offset_y = 0.045

        fig = plt.figure()
        ax = plt.subplot(111)
        patch = Rectangle((patch_posx, patch_posy), width=patch_width, height=patch_height, facecolor=(0.8627, 0.8627, 0.8627), edgecolor = 'black', transform=fig.transFigure)

        ax.imshow(f[ylim[0]:ylim[1],xlim[0]:xlim[1],:], vmin = 0, vmax = 1, alpha = 0.8, cmap = 'gray')
        ax.imshow(rgba_image[ylim[0]:ylim[1],xlim[0]:xlim[1],:], cmap = 'jet', vmax = 1, vmin = 0, alpha = alpha)
        
        # Define percentage values for width, height, and positioning of the rectangle.
        width_percentage = 0.20  # 10% of the map width
        height_percentage = 0.01  # 10% of the map height
        position_right_percentage = 0.08  # 5% gap from the right edge of the map
        position_bottom_percentage = 0.02  # 5% gap from the bottom edge of the map

        # Calculate the position and size for the rectangle on the map.
        map_rect_width = (xlim[1] - xlim[0]) * width_percentage
        map_rect_height = (ylim[1] - ylim[0]) * height_percentage

        map_rect_x = xlim[1] - map_rect_width - ((xlim[1] - xlim[0]) * position_right_percentage)
        map_rect_y = ylim[1] - map_rect_height - ((ylim[1] - ylim[0]) * position_bottom_percentage)

        map_rectangle = Rectangle((map_rect_x, map_rect_y), width=map_rect_width, height=map_rect_height, facecolor=('white'), edgecolor='black', alpha = 0.8)
        ax.add_patch(map_rectangle)


        # Calculate the middle position of the rectangle
        middle_x = map_rect_x + map_rect_width / 2
        middle_y = (map_rect_y + map_rect_height / 2) * 0.97

        # Add text in the middle of the rectangle. Adjust the horizontal and vertical alignment to center the text.
        text_label = "2 Km"  
        ax.text(middle_x, middle_y, text_label, horizontalalignment='center', verticalalignment='center', color='black', fontsize=9, alpha = 0.8)


        ax.set_xticks([])
        ax.set_yticks([])

        # # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        fig.patches.append(patch)
    
      

        fig.text(patch_posx + offset_x, patch_posy +  patch_height + 0.2*offset_y, 'Mineral Colours', transform=fig.transFigure, color ='black')
        colors = list(selected_minerals.values())
        names = list(selected_minerals.keys())
        # find the position of each line by dividing the height of the 'box' by the number of lines and some offset for each
        for i in range(num_lines):
            position = (patch_posx + offset_x, patch_posy + patch_height - (i + 1) * offset_y)
            fig.text(position[0], position[1], names[i], transform=fig.transFigure, color = [x/255 for x in colors[i]])

 

        map_name = basemap.split('/')[-1].split('_')[0]
        plt.title(f'Object ID: {map_name}')
        plt.savefig(f'{map_name}.png', bbox_inches='tight', dpi = 400)
        plt.show()
        return rgba_image[ylim[0]:ylim[1],xlim[0]:xlim[1],:]
    

    def return_top_10_minerals(self, compMapName):
        """
            returns the top 10 minerals from the guess map
            based on the mean cosine distance
        """
        map = envi.open(compMapName).load()
        map = np.array(map)

        map_flat = map.reshape(map.shape[0]*map.shape[1], map.shape[2])
        masked = np.ma.masked_equal(map_flat, 0)


        col_means = masked.mean(axis=0).data
        sorted_index = np.argsort(col_means)[::-1]
        top_10 = col_means[sorted_index[:25]]
        top_10_indices = sorted_index[:25]

        return top_10_indices


   
    def optimized_return_selected_mineral_locations(self, mineral_index, map_name, threshold=0.9):
        """
            returns the locations of the selected mineral
        """
        map = envi.open(map_name).load()
        # Find all locations where the maximum value's index matches mineral_index
        locations = np.argwhere((np.argmax(map, axis=-1) == mineral_index) & (map.max(axis=-1) > threshold))
        
        # Convert the locations to a list of lists
        x_y = locations.tolist()
        
        return x_y

    

    def produce_figure_with_spectra(self, guessmap, mineral_used, basemap, colour_matrix = {}):
        """
            Function to produce the figure with the guess map alongside a spectral plot
            with the location of the spectra.
            :param guessmap: The name of the guess map
            :param mineral_used: The minerals used
            :param basemap: The name of the base map
            :param colour_matrix: A dictionary of the colours for the minerals
            :return: The guess map
        """

        spectra = 'smoothed_5.hdr'
        spectra = envi.open(spectra)
        indices = self.optimized_return_selected_mineral_locations(mineral_index = int(mineral_used[0]), map_name= 'smoothed_5_Cosine_compBestGuess.hdr')
        minerals = envi.open('../Tools/exemplars_normalised.hdr').load()
        minerals = minerals.reshape(minerals.shape[0],240)
        'Now plot the guess map'
        mineral_map = envi.open(guessmap)
        mineral_map = mineral_map.load()
        ylim = [0, 480]
        xlim = [27, 600]

        rgb_mineral_map = np.dstack((mineral_map[:,:,0], mineral_map[:,:,1], mineral_map[:,:,2]))

        rgb_mineral_map = rgb_mineral_map.astype(float)/255.0

        transparent_mask = np.all(rgb_mineral_map == [0, 0, 0], axis=-1)
        alpha_channel = np.where(transparent_mask, 0, 1)

        rgba_image = np.dstack((rgb_mineral_map, alpha_channel))


        mat = False_composite().load_image(basemap)
        if_, rem = False_composite().filter_bad_pixels(mat['IF'])
        im_shape = False_composite().image_shape(mat)
        f = False_composite().get_false_colors(if_.reshape(*im_shape, -1), rem.reshape(im_shape))
        f = np.flip(f, axis = (0,1))


        # fig , axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10,4))
        mineral_names = list(colour_matrix.keys())

        import random
        import matplotlib.gridspec as gridspec

        random_ind1 = random.randint(0,len(indices))
        random_ind2 = random.randint(0,len(indices))



        x1 = indices[random_ind1][0]
        y1 = indices[random_ind1][1]
        x2 = indices[random_ind2][0]
        y2 = indices[random_ind2][1]

        fig = plt.figure(figsize=(15, 6))  # Adjust the overall figure size
        bands = self.get_bands()
        # Define the grid for subplots
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])  # Adjust the ratio of the widths

        # Create the bigger subplot for the first image
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(f, vmin=0, vmax=1)
        ax0.imshow(rgba_image, cmap='jet', vmax=1, vmin=0, alpha = 0.9)
        ax0.scatter(y1, x1, marker='x', color='r', s = 100)
        ax0.scatter(y2, x2, marker='x', color='b', s = 100)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title('Mineral Locations')

        # Create the smaller subplot for the second image
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(bands, spectra[x1, y1], color='r')
        ax1.plot(bands, spectra[x2, y2], color='b')
        ax1.grid(True)
        ax1.set_title('Spectra from pixel locations')
        ax1.set_ylabel('I/F (Reflectance)')
        ax1.set_xlabel('Wavelength')

        # Create the smaller subplot for the third image
        ax2 = fig.add_subplot(gs[2])
        ax2.plot(bands, minerals[mineral_used[0]],  color='g')
        ax2.set_title(f'Exemplar: {mineral_names[mineral_used[0]]} Spectra')
        ax2.grid(True)
        ax2.set_xlabel('Wavelength')
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        ax1.set_xlim(1, 2.6)
        ax2.set_xlim(1, 2.6)


        ax0.text(0.05, 0.95, 'A', transform=ax0.transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))
        ax1.text(0.05, 0.95, 'B', transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))
        ax2.text(0.05, 0.95, 'C', transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='right', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))

        plot_dir = 'Mineral_spectra_plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.tight_layout()  
        plt.savefig(f'Mineral_spectra_plots/{mineral_names[mineral_used[0]]}_spectra.png', bbox_inches='tight', dpi=400)
        plt.show()
        return rgba_image[ylim[0]:ylim[1],xlim[0]:xlim[1],:]
    

    def get_bands(self):
        BANDS = np.array([
        1.021, 1.02755, 1.0341, 1.04065, 1.0472, 1.05375, 1.0603, 1.06685,
        1.07341, 1.07996, 1.08651, 1.09307, 1.09962, 1.10617, 1.11273, 1.11928,
        1.12584, 1.13239, 1.13895, 1.14551, 1.15206, 1.15862, 1.16518, 1.17173,
        1.17829, 1.18485, 1.19141, 1.19797, 1.20453, 1.21109, 1.21765, 1.22421,
        1.23077, 1.23733, 1.24389, 1.25045, 1.25701, 1.26357, 1.27014, 1.2767,
        1.28326, 1.28983, 1.29639, 1.30295, 1.30952, 1.31608, 1.32265, 1.32921,
        1.33578, 1.34234, 1.34891, 1.35548, 1.36205, 1.36861, 1.37518, 1.38175,
        1.38832, 1.39489, 1.40145, 1.40802, 1.41459, 1.42116, 1.42773, 1.43431,
        1.44088, 1.44745, 1.45402, 1.46059, 1.46716, 1.47374, 1.48031, 1.48688,
        1.49346, 1.50003, 1.50661, 1.51318, 1.51976, 1.52633, 1.53291, 1.53948,
        1.54606, 1.55264, 1.55921, 1.56579, 1.57237, 1.57895, 1.58552, 1.5921,
        1.59868, 1.60526, 1.61184, 1.61842, 1.625, 1.63158, 1.63816, 1.64474,
        1.65133, 1.65791, 1.66449, 1.67107, 1.67766, 1.68424, 1.69082, 1.69741,
        1.70399, 1.71058, 1.71716, 1.72375, 1.73033, 1.73692, 1.74351, 1.75009,
        1.75668, 1.76327, 1.76985, 1.77644, 1.78303, 1.78962, 1.79621, 1.8028,
        1.80939, 1.81598, 1.82257, 1.82916, 1.83575, 1.84234, 1.84893, 1.85552,
        1.86212, 1.86871, 1.8753, 1.8819, 1.88849, 1.89508, 1.90168, 1.90827,
        1.91487, 1.92146, 1.92806, 1.93465, 1.94125, 1.94785, 1.95444, 1.96104,
        1.96764, 1.97424, 1.98084, 1.98743, 1.99403, 2.00063, 2.00723, 2.01383,
        2.02043, 2.02703, 2.03363, 2.04024, 2.04684, 2.05344, 2.06004, 2.06664,
        2.07325, 2.07985, 2.08645, 2.09306, 2.09966, 2.10627, 2.11287, 2.11948,
        2.12608, 2.13269, 2.1393, 2.1459, 2.15251, 2.15912, 2.16572, 2.17233,
        2.17894, 2.18555, 2.19216, 2.19877, 2.20538, 2.21199, 2.2186, 2.22521,
        2.23182, 2.23843, 2.24504, 2.25165, 2.25827, 2.26488, 2.27149, 2.2781,
        2.28472, 2.29133, 2.29795, 2.30456, 2.31118, 2.31779, 2.32441, 2.33102,
        2.33764, 2.34426, 2.35087, 2.35749, 2.36411, 2.37072, 2.37734, 2.38396,
        2.39058, 2.3972, 2.40382, 2.41044, 2.41706, 2.42368, 2.4303, 2.43692,
        2.44354, 2.45017, 2.45679, 2.46341, 2.47003, 2.47666, 2.48328, 2.4899,
        2.49653, 2.50312, 2.50972, 2.51632, 2.52292, 2.52951, 2.53611, 2.54271,
        2.54931, 2.55591, 2.56251, 2.56911, 2.57571, 2.58231, 2.58891, 2.6
    ])
    
        return (BANDS)