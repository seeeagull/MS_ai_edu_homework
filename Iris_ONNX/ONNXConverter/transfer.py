import onnx
from onnx import helper,numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import argparse
import json
import numpy as np

def ModelTransfer(model_path,output_path):
    with open(model_path,"r") as f:
        model_define=json.load(f)
    node_list=[]
    input_list=[]
    output_list=[]

    size=model_define["0"]["input_size"]
    if len(size)>2:
        size=[size[0],size[3],size[1],size[2]]
    input_list.append(
        helper.make_tensor_value_info(
            model_define["0"]["input_name"],
            TensorProto.FLOAT,
            size
        )
    )
    
    for index in range(len(model_define)):
        node=model_define[str(index)]
        #全连接层
        if node["type"]=="Fc":
            #建立权重矩阵节点
            s=np.load(node["weights_path"]).astype(np.float32)
            node_list.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[node["weights_name"]],
                    value=helper.make_tensor(
                        name=node["weights_name"],
                        data_type=TensorProto.FLOAT,
                        dims=s.shape,
                        vals=s.flatten().astype(float),
                    ),
                )
            )
            #建立偏置矩阵节点
            s=np.load(node["bias_path"]).astype(np.float32)
            node_list.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[node["bias_name"]],
                    value=helper.make_tensor(
                        name=node["bias_name"],
                        data_type=TensorProto.FLOAT,
                        dims=s.shape,
                        vals=s.flatten().astype(float),
                    ),
                )
            )
            #建立乘法节点
            node_list.append(
                helper.make_node(
                    "MatMul",
                    [node["input_name"],node["weights_name"]],
                    [node["output_name"]+"Temp"]
                )
            )
            #建立加法节点
            node_list.append(
                helper.make_node(
                    "Add",
                    [node["output_name"]+"Temp",node["bias_name"]],
                    [node["output_name"]]
                )
            )

        elif node["type"]=="Relu" or node["type"]=="Softmax" or node["type"]=="Sigmoid":
            node_list.append(
                helper.make_node(
                    node["type"],
                    [node["input_name"]],
                    [node["output_name"]]
                )
            )

        elif node["type"]=="Reshape":
            shape=np.array(node["shape"],dtype=np.int64)

            node_list.append(
                helper.make_node(
                    'Transpose',
                    inputs=[node["input_name"]],
                    outputs=[node["input_name"]+"T"],
                    perm=[0,2,3,1]
                )
            )

            node_list.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[node["output_name"]+"shape"],
                    value=helper.make_tensor(
                        name=node["output_name"]+"shape",
                        data_type=TensorProto.INT64,
                        dims=shape.shape,
                        vals=shape.flatten(),
                    ),
                )
            )
            node_list.append(
                helper.make_node(
                    node["type"],
                    [node["input_name"]+"T",node["output_name"]+"shape"],
                    [node["output_name"]],
                )
            )

    size=model_define[str(index)]["output_size"]
    if len(size)>2:
        size=[size[0],size[3],size[1],size[2]]
    output_list.append(
        helper.make_tensor_value_info(
            model_define[str(index)]["output_name"],TensorProto.FLOAT,size
        )
    )

    graph_proto=helper.make_graph(
        node_list,
        "test",
        input_list,
        output_list,
    )

    onnx.checker.check_node(node_list[1])
    onnx.checker.check_graph(graph_proto)

    model_def=helper.make_model(graph_proto,producer_name="test_onnx")
    onnx.checker.check_model(model_def)
    onnx.save(model_def,output_path)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-m","--model_path",help="the path to the model file")
    parser.add_argument("-m", "--model_path", help="the path to the model file")
    parser.add_argument("-o", "--output_path", help="the path to store the output model")
    
    args = parser.parse_args()
    # test()
    # ModelTransfer(args.model_path, args.output_path)
    ModelTransfer("./output/model.json", "./my_iris.onnx")