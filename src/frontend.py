"""

"""

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as MeanIoU

from src.Datagen import DataSequence
from src.backend import ENET, VGG, UNET
from models.m_light_1 import ModelLight1
from models.bilstm_1 import BiLSTM_1
from models.bilstm_unet import UBNet
from models.bilstm_unet_shorter import UBNet_shorter
from models.bilstm_unet_shorter_upsample import UBNet_shorter_upsample
from keras import callbacks
from loss.weighted_loss import wce


class Segment(object):

    def __init__(self, backend, input_size, nb_classes):

        """
        Model Factory that fetches the corresponding model based on the backend that has been defined
        and initiates the training process

        :param backend: define the backbone architecture for the training
        :param input_size: the size of the input image
        :param nb_classes: the number of classes
        """
        self.input_size = input_size
        self.nb_classes = nb_classes

        if backend == "ENET":
            self.feature_extractor = ENET(self.input_size, self.nb_classes).build()
        elif backend == "VGG":
            self.feature_extractor = VGG(self.input_size, self.nb_classes).build()
        elif backend == "UNET":
            self.feature_extractor = UNET(self.input_size, self.nb_classes).build()
        elif backend == "LIGHT1":
            self.feature_extractor = ModelLight1(self.input_size, self.nb_classes).build()
        elif backend == "BiLSTM_1":
            self.feature_extractor = BiLSTM_1(self.input_size, self.nb_classes).build()
        elif backend == "UBNet":
            self.feature_extractor = UBNet(self.input_size, self.nb_classes).build()
        elif backend == "UBNet_shorter":
            self.feature_extractor = UBNet_shorter(self.input_size, self.nb_classes).build()
        elif backend == "UBNet_shorter_upsample":
            self.feature_extractor = UBNet_shorter_upsample(self.input_size, self.nb_classes).build()
        else:
            raise ValueError('No such arch!... Please check the backend in config file')
            
        self.feature_extractor.summary()

    def train(self, train_configs, valid_configs, model_config):

        """
         Train the model based on the training configurations
        :param train_configs: Configuration for the training
        """
        if "Adam" in train_configs["optimizer"]:
            optimizer = Adam(train_configs["learning_rate"])
        elif "SGD" in train_configs["optimizer"]:
            optimizer = SGD(lr=train_configs["learning_rate"],
                            decay=train_configs["decay"],
                            momentum=train_configs["momentum"], nesterov=True)
        else:
            raise ValueError('Only Adam and SGD are supported! Please specify one of them.')

        train_times = train_configs["train_times"]

        # Data sequence for training
        seqTrain = DataSequence(train_configs["data_directory"], train_configs["batch_size"],
                                self.input_size, True)
        seqValid = DataSequence(valid_configs["data_directory"], valid_configs["batch_size"],
                                self.input_size, False)
        steps_per_epoch = len(seqTrain) * train_times
        
        print("*****************[TRAINING INFO]*****************")
        print("Training set # of batch: "+str(len(seqTrain))+", with batch size: "+str(train_configs["batch_size"]))
        print("Validation set length: "+str(len(seqValid)))
        print("train_times: "+str(train_times))
        print("steps_per_epoch: "+str(steps_per_epoch))
        print("*************************************************")

        if train_configs["loss"] == "categorical_crossentropy":
            loss = 'categorical_crossentropy'
        elif train_configs["loss"] == "wce":
            loss = wce(0.8)
        else:
            raise ValueError('Only categorical_crossentropy and weighted_categorical_crossentropy are supported! '
                             'Please specify one of them.')

        # configure the model for training
        self.feature_extractor.compile(optimizer=optimizer,
                                       loss=loss,
                                       metrics=['acc', 'mae'])
                                      # metrics=['mae'])

        # define the callbacks for training
        tb = TensorBoard(log_dir=train_configs["logs_dir"], write_graph=True)
        mc = ModelCheckpoint(mode='max',
                             filepath='result/'+model_config["backend"]+train_configs["save_model_name"],
                             monitor='val_acc',
                             save_best_only='True',
                             save_weights_only='True', verbose=2)
        es = EarlyStopping(mode='max', monitor='val_acc', patience=20, verbose=1)
        model_reducelr = ReduceLROnPlateau(
            monitor='val_acc',
            factor=0.2,
            patience=10,
            verbose=1,
            min_lr=0.01 * train_configs["learning_rate"])

        callback = [tb, mc, model_reducelr]

        # Train the model on data generated batch-by-batch by the DataSequence generator
        self.feature_extractor.fit_generator(seqTrain,
                                             validation_data=seqValid,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=train_configs["nb_epochs"],
                                             verbose=1,
                                             shuffle=True, callbacks=callback,
                                             workers=3,
                                             max_queue_size=8)
