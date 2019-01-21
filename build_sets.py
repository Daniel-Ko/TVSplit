from os import listdir, chdir

from shutil import copy, rmtree
from pathlib import Path

from time import time

import send2trash

from sklearn.model_selection import train_test_split

N_TRAIN: int, N_VALID: int = 3600, 901

class Splitter:
    self._original_datapath: Path   = Path().resolve() / "data" / "orig_data" 
    self._training_datapath: Path   = Path().resolve() / "data" / "train_data"
    self._validation_datapath: Path = Path().resolve() / "data" / "validation_data"
    self._test_datapath: Path       = Path().resolve() / "data" / "test_data"

    def __init__(self, o_path=self._original_datapath, 
    train_path=self._training_datapath, 
    valid_path=self._validation_datapath, 
    test_path=self._test_datapath):
        ...

""" Deterministically recreate the training and validation sets from the original source.
    
    The original image set must exist in the 'data/orig_data' path, and be separated by class names.

    Params:
        gdrive (bool): True change dataset paths to be local to the Colab environ

"""
def split(n_train, n_valid, seed, train_datapath=TRAIN_DATA_PATH, valid_datapath=VALID_DATA_PATH, test_datapath=TEST_DATA_PATH):
    print("BUILDING DATASETS")
    start = time()

    N_TRAIN, N_VALID = 0, 0
    total_train_imgs, total_valid_imgs = 0, 0

    # The image categories [cherry, strawberry, tomato]
    categories = listdir(ORIGINAL_DATA_PATH)

    # For each of the categories to classify, split all images into training/validation
    for img_category in categories:

        # From each category, sort it deterministically
        orig_img_fnames: list(str) = listdir(ORIGINAL_DATA_PATH / img_category)
        orig_img_fnames.sort()

        train_data, validation_data = train_test_split(
            orig_img_fnames, train_size=0.80, random_state=SEED, shuffle=True
        )

        # Initialise the directory as a Path()
        train_dest = TRAIN_DATA_PATH / img_category
        valid_dest = VALID_DATA_PATH / img_category

        # Either create directory on first call, or empty out the directory if rebuilding
        create_empty_dir(train_dest, gdrive)
        create_empty_dir(valid_dest, gdrive)

        # Copy each the images to their respective train/valid directories
        for train_img in train_data:
            copy(ORIGINAL_DATA_PATH / img_category / train_img, train_dest)
        for valid_img in validation_data:
            copy(ORIGINAL_DATA_PATH / img_category / valid_img, valid_dest)

        # Keep track of how many images were copied
        total_train_imgs += len(train_data)
        total_valid_imgs += len(validation_data)

    print("=========================================================")
    print("DATASETS CREATED SUCCESSFULLY")
    print(f"Successfully created a training set of size: {total_train_imgs}")
    print(f"Successfully created a validation set of size: {total_valid_imgs}")
    print(f"Took {time()-start:.3f} seconds")
    print("=========================================================")

    # Finally, set the number of total images
    N_TRAIN, N_VALID = total_train_imgs, total_valid_imgs




def create_empty_dir(pathobj, gdrive):
    if not pathobj.exists():
        pathobj.mkdir(parents=True)
    else:
        if gdrive:
            rmtree(pathobj)
        else:
            send2trash.send2trash(str(pathobj))
        pathobj.mkdir(parents=True)
