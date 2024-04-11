# AMLSII_23-24_Final Assignment
This project aims at exploring machine learning and deep learning solutions to Image Super-Resolution.

## Code Structure
* [`A`](./A/): Track 1 -- Bicubic Downscaling
    * [`SRGAN.py`](./A/SRGAN.py): Generator and Discriminator of *SRGAN*
    * [`dataset.py`](./A/dataset.py): define dataset Div2kDataset
    * [`loss.py`](./A/loss.py): define generator loss
    * [`train.py`](./A/train.py): training script implemented with PyTorch
    * [`evaluate.py`](./A/evaluate.py): evaluation script (for both validation and testing)

* [`B`](./B/): Track 2 -- Unknown Downscaling
as the training file is the same as A, only model is defined here.
    * [`model.py`](./B/model.py): model used for Track 2

* [`Datasets`](./Datasets/): data downloaded via https://data.vision.ee.ethz.ch/cvl/DIV2K/
    * [`DIV2K_train_HR`]: HR training data
    * [`DIV2K_valid_HR`]: HR validation data
    * [`DIV2K_test_HR`]: HR testing data
    * [`DIV2K_train_LR_bicubic`]: LR training data for Track 1
    * [`DIV2K_valid_LR_bicubic`]: LR validation data for Track 1
    * [`DIV2K_test_LR_bicubic`]: LR testing data for Track 1
    * [`DIV2K_train_LR_unknown`]: LR training data for Track 2
    * [`DIV2K_valid_LR_unknown`]: LR validation data for Track 2
    * [`DIV2K_test_LR_unknown`]: LR testing data for Track 2

* [`main.py`](./main.py): train the GAN-based model for each track

## Installation and Requirements
The code requires common Python environments for model training:
- Python 3.11.5
- PyTorch==1.3.1
- numpy==1.26.1
- tqdm==4.66.1
