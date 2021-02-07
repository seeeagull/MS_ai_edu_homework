class CLayer(object):
    def __init__(self,layer_tp):
        self.layer_type=layer_tp

    def initialize(self,folder,name):
        pass

    def train(self,input,train=True):
        pass

    def pre_update(self):
        pass

    def update(self):
        pass

    def save_parameters(self):
        pass

    def load_parameters(self):
        pass