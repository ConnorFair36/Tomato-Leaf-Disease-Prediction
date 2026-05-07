from yacs.config import CfgNode as CN

_C = CN()
# what mode to run the system in (train, inference)
_C.MODE = "train"

# settings for the dataset
_C.DATASET = CN()

# number of subprocesses to spawn for each dataset to speed up reading data in
_C.DATASET.WORKERS = 2
# the preprocessor to apply to the dataset
_C.DATASET.PREPROCESSOR = "default"

# settigns for model training
_C.TRAINING = CN()

# number of epochs to run and save model weights from
_C.TRAINING.EPOCHS = [10, 25, 50]
# the optimizer to use: [SGDmomentum, Adam, AdamW]
_C.TRAINING.OPTIMIZER = "SGDmomentum"
# the learning rate for the model
_C.TRAINING.LEARNING_RATE = 1e-4
# the name of the model that will be used for saving weights and training data
_C.TRAINING.MODEL_NAME = "CNN_SGD"


# settings for inferencing a model
_C.INFERENCE = CN()

# model weights to try out
_C.INFERENCE.MODEL_WEIGHTS = "src/weights/CNN_model.pth"
# which dataset to run inference on (validation, test)
_C.INFERENCE.DATASET = "test"

config = _C
