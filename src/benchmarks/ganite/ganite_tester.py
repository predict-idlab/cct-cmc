# Copied from: https://github.com/toonvds/NOFLITE

import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ganite.ganite import ganite
from benchmarks.ganite.ganitelin import ganitelin
from data_loading import IHDPData
from metrics import calculatePEHE, calculateATE
from pytorch_lightning.loggers import WandbLogger


class ganite_tester:
    def __init__(self, IHDPData, params, logger, linear):
        """
        Class in charge of initializing, training & testing the Ganite model to be used as benchmark for this project
        :param IHDPData: Dataset (can also be twins dataset
        :param params: initialization parameters
        """
        # Set parameters for training
        self.logger = logger
        self.parameters = dict()
        self.parameters['h_dim'] = 8
        self.parameters['batch_size'] = 128 #params.get("batchSize")
        self.parameters['alpha'] = 2
        self.parameters['iteration'] = 100 # 100 for twins
        self.linear = linear

        # Get data
        # self.data = np.loadtxt('datasets/ihdp_npci_1.txt', delimiter=",", skiprows=1)
        self.XTrain, self.YTrain, self.TTrain = IHDPData().getTrainData()
        self.XTest, self.YTest, self.TTest = IHDPData().getTestData()


    def ganite_train(self):
        """
        Train the Ganite model, returns estimates on the test set
        """
        # Train & get observations
        if self.linear:
            self.y_test_hat = ganitelin(self.XTrain, self.TTrain, self.YTrain, self.XTest, self.parameters)
        else:
            self.y_test_hat = ganite(self.XTrain, self.TTrain, self.YTrain, self.XTest, self.parameters)

    def evaluateModel(self):
        """
        Calcualte PEHE as metric
        """
        # Retrieve test data
        self.pehe = calculatePEHE(self.YTest, self.y_test_hat)
        self.ate = calculateATE(self.YTest, self.y_test_hat)
        dict ={
            'GANITE/Train/PEHE1': self.pehe
        }
        self.logger.log_metrics(dict)
        print('Ganite pehe:', self.pehe)

    def getPEHE(self):
        """
        PEHE getter
        :return: PEHE
        """
        return self.pehe

    def getATE(self):
        return self.ate


wandlogger = WandbLogger(project='benchmark')

ganiteTester = ganite_tester(IHDPData=IHDPData, params={}, logger=wandlogger, linear=False)
ganiteTester.ganite_train()
ganiteTester.evaluateModel()
result = ganiteTester.getPEHE()
print(result)
