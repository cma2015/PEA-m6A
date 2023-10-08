from src.weaknets import WeakRM, DenseClassifier, WeakMASS,WeakRMFine
from src.config import *

class SingleWeakRM:
    def __init__(self, training = True):
        self.extractor = WeakRM(training)
        self.classifer = DenseClassifier()

class SingleReWeakRM:
    def __init__(self, training = True):
        self.extractor = WeakMASS(training)
        self.classifer = DenseClassifier()

class SingleWeakRMFine:
    def __init__(self, training = True):
        self.extractor = WeakRMFine(training)
        self.classifer = DenseClassifier()