# ===============================================================
# import stuff
# ===============================================================
import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import glob
import zipfile, re
import utils
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5


# ===============================================================
# ===============================================================
def get_image_and_label_paths(filename,
                              protocol,
                              extraction_folder):
    # ========================
    # read the contents inside the top-level subject directory
    # ========================
    with zipfile.ZipFile(filename, 'r') as zfile:
        # ========================
        # search for the relevant files
        # ========================
        for name in zfile.namelist():
            # ========================
            # search for files inside the T1w directory
            # ========================
            if re.search(r'\/T1w/', name) != None:
                # ========================
                # search for .gz files inside the T1w directory
                # ========================
                if re.search(r'\.gz$', name) != None:
                    # ========================
                    # get the protocol image
                    # ========================
                    if re.search(protocol + 'w_acpc_dc_restore_brain', name) != None:
                        _imgpath = zfile.extract(name,
                                                 extraction_folder)  # extract the image filepath
                        _patname = name[:name.find('/')]  # extract the patient name

                    # ========================
                    # get the segmentation mask
                    # ========================
                    if re.search('aparc.aseg', name) != None:  # segmentation mask with ~100 classes
                        if re.search('T1wDividedByT2w_', name) == None:
                            _segpath = zfile.extract(name,
                                                     extraction_folder)  # extract the segmentation mask

    return _patname, _imgpath, _segpath


# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 output_file,
                 idx_start,
                 idx_end,
                 protocol,
                 size,
                 target_resolution,
                 preprocessing_folder):
    # ========================
    # read the filenames
    # ========================
    filenames = sorted(glob.glob(input_folder + '*.zip'))
    logging.info('Number of images in the dataset: %s' % str(len(filenames)))

    # =======================
    # create a hdf5 file
    # =======================
    # hdf5_file = h5py.File(output_file, "w")
    #
    # # ===============================
    # # Create datasets for images and labels
    # # ===============================
    # data = {}
    # num_subjects = idx_end - idx_start
    #
    # data['images'] = hdf5_file.create_dataset("images", [num_subjects] + list(size), dtype=np.float32)
    # data['labels'] = hdf5_file.create_dataset("labels", [num_subjects] + list(size), dtype=np.uint8)
    #
    # # ===============================
    # initialize lists
    # ===============================
    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []

    # ===============================
    # initiate counter
    # ===============================
    patient_counter = 0

    # ===============================
    # iterate through the requested indices
    # ===============================
    for idx in range(idx_start, idx_end):
        logging.info('Volume {} of {}...'.format(idx, idx_end))

        # ==================
        # get file paths
        # ==================
        patient_name, image_path, label_path = get_image_and_label_paths(filenames[idx],
                                                                         protocol,
                                                                         preprocessing_folder)

        # ============
        # read the image and normalize it to be between 0 and 1
        # ============
        image, _, image_hdr = utils.load_nii(image_path)

        # ==================
        # read the label file
        # ==================
        label, _, _ = utils.load_nii(label_path)
        label = utils.group_segmentation_classes(label)  # group the segmentation classes as required

        # # ==================
        # # collect some header info.
        # # ==================
        # px_list.append(float(image_hdr.get_zooms()[0]))
        # py_list.append(float(image_hdr.get_zooms()[1]))
        # pz_list.append(float(image_hdr.get_zooms()[2]))
        # nx_list.append(image.shape[0])
        # ny_list.append(image.shape[1])
        # nz_list.append(image.shape[2])
        # pat_names_list.append(patient_name)

        # ==================
        # crop volume along all axes from the ends (as there are several zeros towards the ends)
        # ==================
        image = utils.crop_or_pad_volume_to_size_along_x(image, 256)
        label = utils.crop_or_pad_volume_to_size_along_x(label, 256)
        image = utils.crop_or_pad_volume_to_size_along_y(image, 256)
        label = utils.crop_or_pad_volume_to_size_along_y(label, 256)
        image = utils.crop_or_pad_volume_to_size_along_z(image, 256)
        label = utils.crop_or_pad_volume_to_size_along_z(label, 256)

        # ==================
        # normalize the image
        # ==================
        image_normalized = utils.normalise_image(image, norm_type='div_by_max')

        # ======================================================
        # rescale, crop / pad to make all images of the required size and resolution
        # ======================================================
        scale_vector = [image_hdr.get_zooms()[0] / target_resolution[0],
                        image_hdr.get_zooms()[1] / target_resolution[1],
                        image_hdr.get_zooms()[2] / target_resolution[2]]

        image_rescaled = transform.rescale(image_normalized,
                                           scale_vector,
                                           order=1,
                                           preserve_range=True,
                                           multichannel=False,
                                           mode='constant')

        # label_onehot = utils.make_onehot(label, nlabels=15)
        #
        # label_onehot_rescaled = transform.rescale(label_onehot,
        #                                           scale_vector,
        #                                           order=1,
        #                                           preserve_range=True,
        #                                           multichannel=True,
        #                                           mode='constant')
        #
        # label_rescaled = np.argmax(label_onehot_rescaled, axis=-1)
        #
        # # ============
        # # the images and labels have been rescaled to the desired resolution.
        # # write them to the hdf5 file now.
        # # ============
        # image_list.append(image_rescaled)
        # label_list.append(label_rescaled)

        # ============
        # write to file
        # ============
        # image_rescaled
        volume_dir = os.path.join(preprocessing_folder, 'volume_{:06d}'.format(idx))
        os.makedirs(volume_dir, exist_ok=True)
        for i in range(size[1]):
            slice_path = os.path.join(volume_dir, 'slice_{:06d}.jpeg'.format(i))
            slice = image_rescaled[:, i, :] * 255
            image = Image.fromarray(slice.astype(np.uint8))
            image.save(slice_path)

    #
    #     _write_range_to_hdf5(data,
    #                          image_list,
    #                          label_list,
    #                          patient_counter,
    #                          patient_counter+1)
    #
    #     _release_tmp_memory(image_list,
    #                         label_list)
    #
    #     # update counter
    #     patient_counter += 1
    #
    # # Write the small datasets
    # hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    # hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    # hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    # hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    # hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    # hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    # hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S10"))
    #
    # # After test train loop:
    # hdf5_file.close()


# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):
    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)
    lab_arr = np.asarray(mask_list, dtype=np.uint8)

    hdf5_data['images'][counter_from: counter_to, ...] = img_arr
    hdf5_data['labels'][counter_from: counter_to, ...] = lab_arr


# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list,
                        mask_list):
    img_list.clear()
    mask_list.clear()
    gc.collect()


# ===============================================================
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                idx_start,
                                idx_end,
                                protocol,
                                size,
                                target_resolution,
                                force_overwrite=False):
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_%s_3d_size_%s_res_%s_from_%d_to_%d.hdf5' % (protocol, size_str, res_str, idx_start, idx_end)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     idx_start,
                     idx_end,
                     protocol,
                     size,
                     target_resolution,
                     preprocessing_folder)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


# ===============================================================
# ===============================================================
if __name__ == '__main__':
    # SET PATHS HERE
    input_folder = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/HCP/3T_Structurals_Preprocessed/'
    preprocessing_folder = '/scratch_net/biwinator02/voanna/gpflow-data/hcp_preproc/MRI_slices_64_isotropic'

    data_hcp = load_and_maybe_process_data(input_folder,
                                           preprocessing_folder,
                                           idx_start=0,
                                           idx_end=1040,
                                           protocol='T1',
                                           size=(64, 64, 64),
                                           target_resolution=(2.8, 2.8, 2.8),
                                           force_overwrite=False)
