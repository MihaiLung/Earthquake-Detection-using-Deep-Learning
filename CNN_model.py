from LayerGenerator import *

class CNN_model(nn.Module):
    # Define entities containing model weights in the constructor.
    def __init__(self):
        super().__init__()

        unit = 150000

        self.final_fcnn_input_size = 0

        # Short window CNN->LSTM
        generator_dict = {
            "CNN": {
                "out_channels": [300, 300, 300],
                "kernel_sizes": [100, 1, 1],
                "strides": [100, 1, 1],
                "pool_kernel_sizes": [0, 0, 0],
                "pool_strides": None,
                "batch_norms": [True, True, True],
                "hierarchical_features": False,
                "activation": nn.ReLU()
            },
            "LSTM": {
                "hidden_size": 300,
                "batch_norm": True,
                "activation": nn.ReLU()
            },
            "FCNN": {
                "hidden_sizes": [200],
                "batch_norms": [True],
                "activation": nn.ReLU(),
                "final_activation": True
            }
        }

        self.CNN_LSTM_generator = ModelGenerator(deepcopy(generator_dict))

        self.CNN_LSTM_norm = self.CNN_LSTM_generator()
        self.CNN_LSTM_diff = self.CNN_LSTM_generator(unit - 1)
        self.CNN_LSTM_int = self.CNN_LSTM_generator()

        self.final_fcnn_input_size += generator_dict["FCNN"]["hidden_sizes"][
                                          -1] * self.CNN_LSTM_generator.models_generated

        # Long window CNN->LSTM
        generator_dict = {
            "CNN": {
                "out_channels": [300, 300, 300],
                "kernel_sizes": [1000, 1, 1],
                "strides": [1000, 1, 1],
                "pool_kernel_sizes": [0, 0, 0],
                "pool_strides": None,
                "batch_norms": [True, True, True],
                "hierarchical_features": False,
                "activation": nn.ReLU()
            },
            "LSTM": {
                "hidden_size": 300,
                "batch_norm": True,
                "activation": nn.ReLU()
            },
            "FCNN": {
                "hidden_sizes": [200],
                "batch_norms": [True],
                "activation": nn.ReLU(),
                "final_activation": True
            }
        }

        self.CNN_LSTM_generator_long = ModelGenerator(deepcopy(generator_dict))

        self.CNN_LSTM_norm_long = self.CNN_LSTM_generator_long()
        self.CNN_LSTM_diff_long = self.CNN_LSTM_generator_long(unit - 1)
        self.CNN_LSTM_int_long = self.CNN_LSTM_generator_long()

        self.final_fcnn_input_size += generator_dict["FCNN"]["hidden_sizes"][
                                          -1] * self.CNN_LSTM_generator_long.models_generated

        # Hierarchical CNN
        generator_dict = {
            "CNN": {
                "out_channels": [500, 400, 300],
                "kernel_sizes": [100, 5, 5],
                "strides": [100, 2, 2],
                "pool_kernel_sizes": [2, 2, 2],
                "pool_strides": None,
                "batch_norms": [True, True, True],
                "hierarchical_features": True,
                "activation": nn.ReLU()
            },
            "FCNN": {
                "hidden_sizes": [200],
                "batch_norms": [True],
                "activation": nn.ReLU(),
                "final_activation": True
            }
        }

        self.CNN_hierarchical_generator = ModelGenerator(deepcopy(generator_dict))

        self.CNN_norm = self.CNN_hierarchical_generator()
        self.CNN_diff = self.CNN_hierarchical_generator(unit - 1)
        self.CNN_int = self.CNN_hierarchical_generator()

        self.final_fcnn_input_size += generator_dict["FCNN"]["hidden_sizes"][
                                          -1] * self.CNN_hierarchical_generator.models_generated

        # STD LSTM
        self.std_window = 1000
        self.std_stride = 500
        self.std_input_size = update_kernel_size(unit, {"kernel_size": self.std_window,
                                                        "stride": self.std_stride})
        # With current set up it's 299 features
        generator_dict = {
            "LSTM": {
                "hidden_size": 30,
                "batch_norm": True,
                "activation": nn.ReLU()
            }
        }

        self.std_LSTM_generator = ModelGenerator(deepcopy(generator_dict))

        self.LSTM_std_norm = self.std_LSTM_generator()
        self.LSTM_std_diff = self.std_LSTM_generator()
        self.LSTM_std_int = self.std_LSTM_generator()

        self.final_fcnn_input_size += generator_dict["LSTM"][
                                          "hidden_size"] * 2 * self.std_LSTM_generator.models_generated

        # 3. Final FCNN module generator

        FCNN_gen_dict = {
            "hidden_sizes": [100, 1],
            "batch_norms": [True, False],
            "activation": nn.ReLU(),
            "final_activation": True
        }
        final_fcnn_generator = FCNNGenerator(deepcopy(FCNN_gen_dict))

        self.final_fcnn = final_fcnn_generator(self.final_fcnn_input_size)

    def forward(self, inputs):
        # Generate processed inputs
        inputs_diff = inputs[:, :, 1:] - inputs[:, :, :-1]
        inputs_int = inputs.cumsum(dim=0)

        inputs_std = inputs.unfold(-1, self.std_window, self.std_stride).std(-1)
        inputs_diff_std = inputs_diff.unfold(-1, self.std_window, self.std_stride).std(-1)
        inputs_int_std = inputs_int.unfold(-1, self.std_window, self.std_stride).std(-1)

        # CNN->LSTM, short window
        h_norm = self.CNN_LSTM_norm(inputs)
        h_diff = self.CNN_LSTM_diff(inputs_diff)
        h_int = self.CNN_LSTM_int(inputs_int)

        fcnn_input = torch.cat((h_norm, h_diff, h_int), dim=1)
        # fcnn_input = torch.cat((h_norm, h_diff), dim=1)

        # CNN->LSTM, long window
        h_norm_long = self.CNN_LSTM_norm_long(inputs)
        h_diff_long = self.CNN_LSTM_diff_long(inputs_diff)
        h_int_long = self.CNN_LSTM_int_long(inputs_int)

        fcnn_input = torch.cat((fcnn_input, h_norm_long, h_diff_long, h_int_long), dim=1)
        # fcnn_input = torch.cat((fcnn_input, h_norm_long, h_diff_long), dim = 1)

        # CNN->hierarchical
        h_norm_cnn = self.CNN_norm(inputs)
        h_diff_cnn = self.CNN_diff(inputs_diff)
        h_int_cnn = self.CNN_int(inputs_int)

        fcnn_input = torch.cat((fcnn_input, h_norm_cnn, h_diff_cnn, h_int_cnn), dim=1)

        # std->LSTM
        h_norm_std = self.LSTM_std_norm(inputs_std)
        h_diff_std = self.LSTM_std_diff(inputs_diff_std)
        h_int_std = self.LSTM_std_int(inputs_int_std)

        fcnn_input = torch.cat((fcnn_input, h_norm_std, h_diff_std, h_int_std), dim=1)

        # FCNN to output

        output = self.final_fcnn(fcnn_input)

        return output

if __name__ == "__main__":
    model = CNN_model()
    print(model)