
from Ratio_tools.train import feat_masks, train_model_bland, train_model, \
    iteration_weights
from spectral.io import envi
from Ratio_tools.io import load_image
from Ratio_tools.preprocessing import filter_bad_pixels
import matplotlib.pyplot as plt
from Ratio_tools.io import image_shape
from Ratio_tools.plot import get_false_colors

import numpy as np
from Ratio_tools.preprocessing import remove_spikes_column, replace, ratio
from Ratio_tools.train import compute_bland_scores

class Ratio_basemap():
    def __init__(self):
        pass


    def create_ratio_img(img_name, flip = False):
        """
        Below function and all inherited modules
        has been taken from the CRISM ML tutorial
        notebook:
        https://github.com/Banus/crism_ml

        Create a ratio image from a given image
        :param img_name: name of the image
        :return: ratio image
        """
        DATADIR = 'Ratio_tools/Datasets'

        fin0, fin = feat_masks()
        bmodels = train_model_bland(DATADIR, fin0)
        models = train_model(DATADIR, fin)
        ww_ = iteration_weights(models[0].classes)

        im_path = 'Basemaps/'
        im_path = im_path + img_name
        mat = load_image(im_path)
        if_, rem = filter_bad_pixels(mat['IF'])

        im_shape = image_shape(mat)

        
        if1 = remove_spikes_column(
            if_.reshape(*im_shape, -1), 3, 5).reshape(if_.shape)
        slog = compute_bland_scores(if1, (bmodels, fin0))

        slog_inf = replace(slog, rem, -np.inf).reshape(im_shape)
        if2 = ratio(if1.reshape(*im_shape, -1), slog_inf).reshape(if_.shape)

        


        # reshape if2 to im_shape
        if2_reshape = if2.reshape(im_shape[0],im_shape[1], 248)
        if flip:
            if2_reshape = np.flip(if2_reshape, axis=(0,1))


        base = 'Ratio_hdrs/'
        paths = f'{img_name[:-4]}.hdr'
        path_to_save = base + paths

        envi.save_image(path_to_save, if2_reshape, dtype=np.float32, force=True,
                                interleave='bil')
        
        return paths, im_shape


