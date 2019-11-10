TRAIN_IMAGE_LOCATION = '../dataset/images/render'
TRAIN_MASK_LOCATION = '../dataset/images/ground'
IMAGE_SIZE = 512
NUM_CLASSES = 32
BATCH_SIZE = 8
EPOCHS = 100
LOSS_HYPERPARAMETERS = {
    'weighted_cross_entropy_beta' : 1.0,
    'balanced_cross_entropy_beta' : 1.0,
    'focal_loss_alpha' : 0.25,
    'focal_loss_gamma' : 2,
    'tversky_loss_beta' : 1.0
}