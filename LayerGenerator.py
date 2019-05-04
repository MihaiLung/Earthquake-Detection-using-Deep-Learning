import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


def update_kernel_size(current_kernel_size: int, d):
    """Input the current kernel size as the first argument, and a dictionary
    which contains the kernel_size and stride arguments of the last
    convolution or pooling layer.
    Returns the updated kernel size"""
    new_kernel_size: int = int((current_kernel_size - d["kernel_size"]) / d["stride"] + 1)
    return new_kernel_size


class CNNandLSTMModuleGenerator:
    def __init__(self, CNN_dict, LSTM_dict = None, input_feats: int = 1):
        """
        Input_feats = dimensions of input features
        CNN_dict should be ordered and should have, for each layer to be generated:
         - out_channels = output channels
         - kernel_size = kernel size
         - stride = stride
         - activation
         - pool = dictionary for pooling layer following this later, including:
            - include = boolean on whether the pooling layer is to be included
            - total = boolean whether to collapse all features to one or not
            - kernel_size - only used if total is False.
            - stride - only used if total is False.
         - batch_norm = boolean on whether the pooling layer is to be included
        LSTM_dict should contain the following values:
         - include = boolean on whether an LSTM layer is to be included or not
         - hidden_size = size of the hidden layer
         - batch_norm = boolean indicating whether batch normalization is to be applied
         on the final outputs of the LSTM layer
        """

        self.CNN_dict = CNN_dict
        self.LSTM_dict = LSTM_dict
        self.input_feats = input_feats

    def __call__(self, input_size: int = 150000, channels: int = 1):
        modules_list = []
        input_feats = self.input_feats
        self.kernel_size = input_size
        if self.CNN_dict is not None:
            for i, key in enumerate(self.CNN_dict):
                # Initialise dict of tasks for this layer
                layer_settings_dictionary = self.CNN_dict[key]

                # Generate convolutional layer
                conv_layer = nn.Conv1d(
                    in_channels=input_feats,
                    out_channels=layer_settings_dictionary["out_channels"],
                    kernel_size=layer_settings_dictionary["kernel_size"],
                    stride=layer_settings_dictionary["stride"]
                )

                # Update input feats, modules, k_size, add activation
                input_feats = layer_settings_dictionary["out_channels"]
                modules_list.extend([conv_layer, layer_settings_dictionary["activation"]])
                self.kernel_size = update_kernel_size(self.kernel_size, layer_settings_dictionary)

                # Check and generate pool module if neede
                if layer_settings_dictionary["pool"]["include"]:
                    if layer_settings_dictionary["pool"]["total"]:
                        # Generate module
                        pool = nn.MaxPool1d(
                            kernel_size=self.kernel_size
                        )
                        # Update modules
                        modules_list.append(pool)
                        self.kernel_size = 1
                    else:
                        # Generate module
                        pool = nn.MaxPool1d(
                            kernel_size=layer_settings_dictionary["pool"]["kernel_size"],
                            stride=layer_settings_dictionary["pool"]["stride"]
                        )
                        # Update modules and k_size
                        modules_list.append(pool)
                        self.kernel_size = update_kernel_size(self.kernel_size, layer_settings_dictionary["pool"])

                if layer_settings_dictionary["batch_norm"]:
                    # Generate module
                    norm = nn.BatchNorm1d(
                        num_features=layer_settings_dictionary["out_channels"]
                    )

                    # Update modules and k_size
                    modules_list.append(norm)

            CNN_module = nn.Sequential(*modules_list)

            if self.LSTM_dict is None:
                return CNN_module
            else:
                class BlockModuleCNNthenLSTM(torch.nn.Module):
                    def __init__(self, LSTM_dict, kernel_size):
                        super(BlockModuleCNNthenLSTM, self).__init__()
                        self.LSTM = LSTM_dict
                        self.CNN_feats = CNN_module
                        self.LSTM_layer = nn.LSTM(
                            input_size=layer_settings_dictionary["out_channels"],
                            hidden_size=self.LSTM["hidden_size"],
                            bidirectional=True,
                            batch_first=True
                        )
                        self.kernel_size = kernel_size

                    def forward(self, x):
                        """
                        In the forward function we accept a Tensor of input data and we must return
                        a Tensor of output data. We can use Modules defined in the constructor as
                        well as arbitrary operators on Tensors.
                        """

                        h = self.CNN_feats(x)
                        h = torch.transpose(h, 1, 2)
                        o, _ = self.LSTM_layer(h)
                        o = o.view(-1, self.kernel_size, 2, self.LSTM["hidden_size"])
                        o = torch.cat((o[:, -1, 0, :], o[:, 0, 1, :]), dim=1)

                        return o.squeeze()

                return BlockModuleCNNthenLSTM(self.LSTM_dict, self.kernel_size)

        else:
            class LSTMCustomForward(torch.nn.Module):

                def __init__(self, hidden_size, channels: int =1):
                    super(LSTMCustomForward, self).__init__()
                    self.input_size = channels
                    self.hidden_size = hidden_size
                    self.LSTM_layer = nn.LSTM(
                        input_size=channels,
                        hidden_size=hidden_size,
                        bidirectional=True,
                        batch_first=True
                    )

                def forward(self, x):
                    o, _ = self.LSTM_layer(x.transpose(1,2))
                    o = o.view(-1, x.shape[-1], 2, self.hidden_size)
                    o = torch.cat((o[:, -1, 0, :], o[:, 0, 1, :]), dim=1)
                    return o

            return LSTMCustomForward(self.LSTM_dict["hidden_size"], channels)


class FCNNGenerator:
    def __init__(self, gen_dict_FCNN):
        # List of hidden dimensions
        self.hidden_sizes = gen_dict_FCNN["hidden_sizes"]

        # PyTorch layer for non-linear activation
        self.activation = gen_dict_FCNN["activation"]

        # List of booleans indicating whether batch normalization is to be applied after the current layer or not
        self.batch_norm = gen_dict_FCNN["batch_norms"]

        self.final_activation = gen_dict_FCNN["final_activation"]

    def __call__(self, in_features: int):
        modules = []
        for i, (dim, norm) in enumerate(zip(self.hidden_sizes, self.batch_norm)):
            module = nn.Linear(
                in_features=in_features,
                out_features=dim
            )
            in_features = dim
            modules.append(module)

            if i+1 < len(self.hidden_sizes) or self.final_activation:
                modules.append(self.activation)

            if norm:
                norm = nn.BatchNorm1d(
                    num_features=dim
                )
                modules.append(norm)

        return nn.Sequential(*modules)


class FullModel(torch.nn.Module):
    """Creates a single module which extracts features using a CNN, optionally passes them to an LSTM, then passes the
    resulting vector to an FCNN.
    The output of the CNN/LSTM MUST be a (batch, dimension) matrix, otherwise awful things shalth happen"""
    def __init__(self, CNN_LSTM_module, FCNN_module):
        super(FullModel, self).__init__()
        self.CNN_LSTM_module = CNN_LSTM_module
        self.FCNN_module = FCNN_module

    def forward(self, x):
        h = self.CNN_LSTM_module(x).squeeze()
        output = self.FCNN_module(h)
        return output


class ModelGenerator:
    """
    Returns an end-to-end CNN, optionally followed by an LSTM, which takes in the raw input and returns a single
    prediction of the desires dimension
    """

    def __init__(self, dict_CNN_LSTM):

        self.models_generated = 0
        self.generator_dict = {}
        # Generate LSTM
        if "CNN" in dict_CNN_LSTM:
            current_dict = dict_CNN_LSTM["CNN"]
            total_poolings = [False] * len(current_dict["out_channels"])

            if current_dict["hierarchical_features"]:
                total_poolings[-1] = True
            if current_dict["pool_strides"] is None:
                current_dict["pool_strides"] = current_dict["pool_kernel_sizes"]

            gen_dict_CNN = OrderedDict()
            for i, (out_channel, kernel_size, stride, pool_kernel_size, pool_stride, batch_norm, total_pooling) in enumerate(
                    zip(current_dict["out_channels"], current_dict["kernel_sizes"], current_dict["strides"],
                    current_dict["pool_kernel_sizes"], current_dict["pool_strides"], current_dict["batch_norms"],
                    total_poolings)):
                layer_dict = {
                    "out_channels": out_channel,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "activation": current_dict["activation"],
                    "pool": {
                        "include": pool_kernel_size>1 or total_pooling,
                        "total": total_pooling,
                        "kernel_size": pool_kernel_size,
                        "stride": pool_stride
                    },
                    "batch_norm": current_dict["batch_norms"]
                }
                gen_dict_CNN["layer_"+str(i+1)] = deepcopy(layer_dict)

            self.fcnn_input_size = current_dict["out_channels"][-1]
            #self.generator_dict["CNN"] = gen_dict_CNN
        else:
            gen_dict_CNN = None

        if "LSTM" in dict_CNN_LSTM:
            gen_dict_LSTM = dict_CNN_LSTM["LSTM"]
            self.fcnn_input_size = gen_dict_LSTM["hidden_size"]*2
        else:
            gen_dict_LSTM = None

        self.CNN_LSTM_generator = CNNandLSTMModuleGenerator(gen_dict_CNN, gen_dict_LSTM)

        if "FCNN" in dict_CNN_LSTM:
            gen_dict_FCNN = dict_CNN_LSTM["FCNN"]
            self.FCNN_generator = FCNNGenerator(gen_dict_FCNN)
        else:
            self.FCNN_generator = None

    def __call__(self, input_size: int = 150000):
        self.models_generated+=1
        CNN_LSTM_module = self.CNN_LSTM_generator(input_size)

        if self.FCNN_generator is not None:
            FCNN_module = self.FCNN_generator(self.fcnn_input_size)
            full_module = FullModel(CNN_LSTM_module,FCNN_module)
            return full_module

        return CNN_LSTM_module


if __name__ == "__main__":
    """
    Includes tests to ensure functionality is as expected
    """
    """
    activation = nn.ReLU()
    layer_dict = {
        "out_channels": 1000,
        "kernel_size": 10,
        "stride": 10,
        "activation": activation,
        "pool": {
            "include": True,
            "total": False,
            "kernel_size": 5,
            "stride": 5
        },
        "batch_norm":  True
    }

    LSTM_dict = {
        "include": False,
        "hidden_size": 100
    }

    CNN_dict = OrderedDict()
    CNN_dict["layer_1"] = deepcopy(layer_dict)

    layer_dict["kernel_size"] = layer_dict["stride"] = 5
    CNN_dict["layer_2"] = deepcopy(layer_dict)

    layer_dict["pool"]["total"] = True
    CNN_dict["layer_3"] = deepcopy(layer_dict)

    generator = CNNModuleGenerator(CNN_dict, deepcopy(LSTM_dict))
    LSTM_dict["include"]=True
    generator_LSTM = CNNModuleGenerator(CNN_dict, deepcopy(LSTM_dict))
    CNN1 = generator.generate()
    #print(network)


    data = torch.randint(-15, 15, (5, 1, 150000)).float()

    hidden_layers = [100,50,10]
    batch_norms = [True,True,False]

    fcnn_generator = FCNNGenerator(hidden_layers,batch_norms,activation)
    fcnn = fcnn_generator.generate(CNN_dict["layer_3"]["out_channels"]*3+LSTM_dict["hidden_size"]*2)


    CNN2 = generator.generate()
    CNN3 = generator.generate()

    LSTMboi = generator_LSTM.generate()

    d_gen = {
        "CNN":{
            "CNN1": CNN1,
            "CNN2": CNN2,
            "CNN3": CNN3
        },
        "LSTM":{
            "LSTMboi": LSTMboi
        },
        "FCNN": fcnn
    }

    prociboi = DataProcessor(d_gen)
    print(prociboi)
    print(prociboi(data).shape)
    """


    dict_CNN_LSTM = {
        "CNN": {
            "out_channels": [50, 50, 50],
            "kernel_sizes": [100, 5, 5],
            "strides": [50, 2, 2],
            "pool_kernel_sizes": [2, 2, -1],
            "pool_strides": None,
            "batch_norms": [True, True, True],
            "hierarchical_features": True,
            "activation": nn.ReLU()
        },
        "FCNN": {
            "hidden_sizes": [50, 10, 2],
            "batch_norms": [True, True, False],
            "activation": nn.ReLU(),
            "final_activation": False
        }
    }

    dict_CNN_LSTM = {
        "CNN": {
            "out_channels": [50, 50, 50],
            "kernel_sizes": [100, 5, 5],
            "strides": [50, 2, 2],
            "pool_kernel_sizes": [2, 2, -1],
            "pool_strides": None,
            "batch_norms": [True, True, True],
            "hierarchical_features": False,
            "activation": nn.ReLU()
        },
        "LSTM": {
            "hidden_size": 100,
            "batch_norm": True,
            "activation": nn.ReLU()
        },
        "FCNN": {
            "hidden_sizes": [50, 10, 2],
            "batch_norms": [True, True, False],
            "activation": nn.ReLU(),
            "final_activation": False
        }
    }

    dict_CNN_LSTM = {
        "LSTM": {
            "hidden_size": 10,
            "batch_norm": True,
            "activation": nn.ReLU()
        }
    }

    generator = ModelGenerator(dict_CNN_LSTM)

    data = torch.randint(-15, 15, (10, 1, 1500)).float()

    data_kinky = data.unfold(2,100,10).std(-1)
    print("Data shape",data_kinky.shape)
    model_kinky = generator(141)
    print(model_kinky(data_kinky).shape)
