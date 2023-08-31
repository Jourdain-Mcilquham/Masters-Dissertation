from Tools.minMapTools import minMapTools
from Tools.modelFitTools import modelFitTools
from Tools.crismProcessing_parallel  import crismProcessing_parallel
from Tools.hsiUtilities import hsiUtilities
from Tools.generalUtilities import generalUtilities
from Tools.GANModel_1d import GANModel_1d
from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
plt.rcParams['font.family'] = 'Arial'
import os

class ROC_functions:
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
        obj = GANModel_1d(img_rows=240, dropout=0.0, genFilters=250, disFilters=20,
                    filterSize=11)
        dis1 = obj.disModel_CV_L6s2()
        'Get the pre-trained weights'
        dis1.load_weights(('dis_cR_67.h5'))
        disRep = obj.disModel_CV_L6s2_rep(dis1)
        
        exemplars = envi.open('exem_library.hdr')
        exemplars = exemplars.load()
        exemplars = exemplars.reshape(exemplars.shape[0], exemplars.shape[1])

        for i in range(len(exemplars)):
            exemplars[i] = (exemplars[i] - np.min(exemplars[i])) / (np.max(exemplars[i]) - np.min(exemplars[i]))

        envi.save_image('exemplars_normalised.hdr', exemplars, dtype='float32', force=True, interleave='bil')
        return  exemplars, disRep
    
    def normalise_img(self, f_name):
        img = envi.open(f_name)
        img = img.load().copy()

        img_normalised = np.zeros((img.shape[0], img.shape[1], 240))

        for i in tqdm(range(img.shape[0]), desc="Normalising Spectra", position=0, leave=True):
            for j in range(img.shape[1]):

                # Normalise img
                img[i,j] = (img[i,j] - np.min(img[i,j])) / (np.max(img[i,j]) - np.min(img[i,j]))
                img[i,j] = savgol_filter(img[i,j], 9,2)
                img_normalised[i,j] = img[i,j][4:244]

        envi.save_image('normalised.hdr', img_normalised, dtype='float32', force=True, interleave='bil')
        return 'normalised.hdr'

    def create_classification_map(self):
        # read in the h5 file and save as list
        name = 'ratio_training2.h5'
        f = h5py.File(name, 'r')
        data = list(f['data'])
        data.shape
        # # Shuffle 
        # # Zipping the two lists together
        # zipped = list(zip(list1, list2))

        # # Shuffling the zipped list
        # random.shuffle(zipped)

        # # Unzipping the shuffled list back into two lists
        # list1_shuffled, list2_shuffled = zip(*zipped)

    def create_map(self, f_name):
        endMem_Name = ''
        # check if f_name directory exists if not create it
        dir_name = f_name[:-4]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        sLvl = 0.1
        normalise_flag = True

   

        exemplars, discriminator = self.GAN_REP()
        os.chdir(dir_name)

        # Normalise the image if needed
        f_name = self.normalise_img(f_name) if normalise_flag else f_name

        'Since we are mapping at different mapping levels'
        for kernelSize in range(1,6,2):
            '--------------------------------------------------------------------------------------------------'
            'STEP-6: SMOOTHING CONTINUUM REMOVED IMAGE'
            '--------------------------------------------------------------------------------------------------'
            crSmoothImageName = modelFitTools(0, 240).crismImgSmooth(f_name, kernelSize)

            '--------------------------------------------------------------------------------------------------'
            'STEP-7: GENERATE SIMILARITY MAPS BETWEEN MICA REPRESENATIONS AND DATA REPRESENTATIONS'
            '--------------------------------------------------------------------------------------------------'
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



    def create_color_map(self, compMapName):
        colour_matrix = {
            'Hematite': [180, 34, 34],
            'Nontronite': [107, 142, 35],
            'Saponite': [124, 252, 0],
            'Prehnite': [102, 205, 170],
            'Jarosite': [255, 223, 0],
            'Serpentine': [50, 205, 50],
            'Alunite': [178, 190, 181],
            'Calcite': [248, 248, 255],
            'Beidellite': [210, 180, 140],
            'Kaolinite': [250, 235, 215],
            'Bassanite': [188, 152, 126],
            'Epidote': [85, 105, 47],
            'Montmorillonite': [160, 82, 45],
            'Mg Cl salt': [238, 232, 170],
            'Halloysite': [255, 228, 181],
            'Epsomite': [240, 248, 255],
            'Illite/Muscovite': [245, 245, 220],
            'Margarite': [255, 228, 196],
            'Analcime': [240, 255, 240],
            'Monohydrated sulfate': [255, 228, 225],
            'MgCO3': [233, 150, 122],
            'Chlorite': [143, 188, 143],
            'Clinochlore': [221, 160, 221],
            'Low Ca Pyroxene': [240, 128, 128],
            'Olivine Forsterite': [238, 130, 238],
            'High Ca Pyroxene': [0, 128, 128],
            'Olivine Fayalite': [0, 100, 0]
        }



        thresholds = {
            'Hematite': [0.85, 0.9],
            'Nontronite': [0.88, 0.9],
            'Saponite': [0.85, 0.9],
            'Prehnite': [0.85, 0.9],
            'Jarosite': [0.85, 0.9],
            'Serpentine': [0.85, 0.9],
            'Alunite': [0.85, 0.9],
            'Calcite': [0.85, 0.9],
            'Beidellite': [0.85, 0.9],
            'Kaolinite': [0.85, 0.9],
            'Bassanite': [0.85, 0.9],
            'Epidote': [0.85, 0.9],
            'Montmorillonite': [0.85, 0.9],
            'Mg Cl salt': [0.85, 0.9],
            'Halloysite': [0.85, 0.9],
            'Epsomite': [0.85, 0.9],
            'Illite/Muscovite': [0.85, 0.9],
            'Margarite': [0.85, 0.9],
            'Analcime': [0.85, 0.9],
            'Monohydrated sulfate': [0.85, 0.9],
            'MgCO3': [0.85, 0.9],
            'Chlorite': [0.85, 0.9],
            'Clinochlore': [0.85, 0.9],
            'Low Ca Pyroxene': [0.85, 0.9],
            'Olivine Forsterite': [0.85, 0.9],
            'High Ca Pyroxene': [0.85, 0.9],
            'Olivine Fayalite': [0.85, 0.9]
        }

        colMat = np.asarray(list(colour_matrix.values()), dtype=np.float32)
        thresholds = np.asarray(list(thresholds.values()), dtype=np.float32)
        # colMat = [x/255 for x in colMat]

        'Generate the Guess map'
        guessMap, minerals = minMapTools(0, 240).create_Maps4CRISMImages_GuessMap(compMapName, colMat, thresholds=thresholds)

        

        return guessMap, minerals

    def produce_figure(self, guessmap, mineral_used, basemap):

        colour_matrix = {
                    'Hematite': [180, 34, 34],
                    'Nontronite': [107, 142, 35],
                    'Saponite': [124, 252, 0],
                    'Prehnite': [102, 205, 170],
                    'Jarosite': [255, 223, 0],
                    'Serpentine': [50, 205, 50],
                    'Alunite': [178, 190, 181],
                    'Calcite': [248, 248, 255],
                    'Beidellite': [210, 180, 140],
                    'Kaolinite': [250, 235, 215],
                    'Bassanite': [188, 152, 126],
                    'Epidote': [85, 105, 47],
                    'Montmorillonite': [160, 82, 45],
                    'Mg Cl salt': [238, 232, 170],
                    'Halloysite': [255, 228, 181],
                    'Epsomite': [240, 248, 255],
                    'Illite/Muscovite': [245, 245, 220],
                    'Margarite': [255, 228, 196],
                    'Analcime': [240, 255, 240],
                    'Monohydrated sulfate': [255, 228, 225],
                    'MgCO3': [233, 150, 122],
                    'Chlorite': [143, 188, 143],
                    'Clinochlore': [221, 160, 221],
                    'Low Ca Pyroxene': [240, 128, 128],
                    'Olivine Forsterite': [238, 130, 238],
                    'High Ca Pyroxene': [0, 128, 128],
                    'Olivine Fayalite': [0, 100, 0] 
                }

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


        mat = False_composite().crism_to_mat(basemap, flatten = True)
        if_, rem = False_composite().filter_bad_pixels(mat['IF'])
        im_shape = False_composite().image_shape(mat)
        f = False_composite().get_false_colors(if_.reshape(*im_shape, -1), rem.reshape(im_shape), channels= (13,18,42))
        f = np.flip(f, axis = (0,1))

        patch_posx = 0.73
        patch_posy = 0.25
        patch_width = 0.25
        patch_height = 0.5
        offset_x = 0.02
        offset_y = 0.045

        print(rgba_image[0])
        fig = plt.figure()
        ax = plt.subplot(111)
        patch = Rectangle((patch_posx, patch_posy), width=patch_width, height=patch_height, facecolor=(0.8627, 0.8627, 0.8627), edgecolor = 'black', transform=fig.transFigure)

        ax.imshow(f[ylim[0]:ylim[1],xlim[0]:xlim[1],:], vmin = 0, vmax = 1)
        ax.imshow(rgba_image[ylim[0]:ylim[1],xlim[0]:xlim[1],:], cmap = 'jet', vmax = 1, vmin = 0)
        ax.set_xticks([])
        ax.set_yticks([])

        # # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        # ax.legend(loc='cen left', bbox_to_anchor=(1, 0.5))
        fig.patches.append(patch)

        text_positions = [(patch_posx + offset_x, patch_posy +  patch_height - 2* offset_y),
                        (patch_posx + offset_x, patch_posy+ patch_height - 3* offset_y), 
                        (patch_posx + offset_x, patch_posy + patch_height - 4 * offset_y)]
       

        # Convert dictionary keys to a list
        keys_list = list(colour_matrix.keys())

        # Create a new dictionary with only the selected keys
        selected_minerals = {keys_list[i]: colour_matrix[keys_list[i]] for i in mineral_used if i < len(keys_list)}

        # Find the number of lines needed to display the minerals
        num_lines = len(selected_minerals)
    

        fig.text(patch_posx + offset_x, patch_posy +  patch_height + 0.2*offset_y, 'Mineral Colours', transform=fig.transFigure, color ='black')
        colors = list(selected_minerals.values())
        names = list(selected_minerals.keys())
        # find the position of each line by dividing the height of the 'box' by the number of lines and some offset for each
        for i in range(num_lines):
            position = (patch_posx + offset_x, patch_posy + patch_height - (i + 1) * offset_y)
            fig.text(position[0], position[1], names[i], transform=fig.transFigure, color = [x/255 for x in colors[i]])


        plt.title('Object ID: FRT000093be')
   
        plt.savefig('legend_outside.png', bbox_inches='tight', dpi = 400)
        plt.show()

