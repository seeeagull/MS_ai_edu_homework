import numpy as np
import os
import json

def model_save(numpy_model,save_path):
    assert type(numpy_model)==list
    os.makedirs(save_path,exist_ok=True)
    model={}
    index=0
    for (index,layer) in enumerate(numpy_model):
        if layer.type=="Fc":
            model[index]={
                "type":"Fc",
                "input_name":layer.input_name,
                "input_size":layer.input_size,
                "weights_name":layer.weights_name,
                "weights_path":os.path.join(save_path,layer.weights_name+".npy"),
                "bias_name":layer.bias_name,
                "bias_path":os.path.join(save_path,layer.bias_name+".npy"),
                "output_name":layer.output_name,
                "output_size":layer.output_size
            }
            np.save(os.path.join(save_path,layer.weights_name+".npy"),layer.W)
            np.save(os.path.join(save_path,layer.bias_name+".npy"),layer.B)

        elif layer.type=="Relu" or layer.type=="Sigmoid" or layer.type=="Softmax":
            model[index]={
                "type":layer.type,
                "input_name":layer.input_name,
                "input_size":layer.input_size,
                "output_name":layer.output_name,
                "output_size":layer.output_size
            }

        elif layer.type=="Reshape":
            model[index]={
                "type":layer.type,
                "input_name":layer.input_name,
                "output_name":layer.output_name,
                "shape":layer.shape
            }

    with open(os.path.join(save_path,"model.json"),"w") as f:
        json.dump(model,f)

    return True