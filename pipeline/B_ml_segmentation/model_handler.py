# SOURCE > https://github.com/UNEP-IMEO-MARS/marsml-hyperspectral
# Author: Vit Ruzicka, 2026
# Class that creates the model given the settings.
import numpy as np
import torch
import segmentation_models_pytorch as smp

def num_of_params(model):
    if model is None: return 0
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class UNET():
    def __init__(self, encoder_name="mobilenet_v2", num_channels=4, num_classes=1, activation=None):
        self.network = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=num_channels, classes=num_classes, activation=activation)

    def forward(self, inputs):
        return self.network(inputs)

    def to(self, device):
        self.network.to(device)


class ModelHandler():
    def __init__(self, settings):
        self.settings = settings

        self.architecture = settings.model.architecture
        
        # probably from settings ...
        self.image_resolution = settings.model.image_resolution
        self.num_channels = settings.model.num_channels
        self.custom_config = None

        # device
        accelerator = settings.trainer.accelerator
        if accelerator == "gpu":
            self.device = torch.device("cuda")
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        elif accelerator == "mps":
            self.device = torch.device("mps")

        self.model = self.initialise_model()

    def initialise_model(self):
        if self.architecture == "unet":
            self.num_classes = 1
            encoder_name = self.settings.model.encoder_name # "mobilenet_v2" or "timm-mobilenetv3_large_100 for example
            model = UNET(encoder_name, num_channels=self.num_channels, num_classes=self.num_classes)

        model.to(self.device)
        return model

    def summarise(self, example_forward=True):
        params = num_of_params(self.model.network)
        M_params = round(params / 1000 / 1000, 2)
        print("[Model] "+self.architecture+" with", str(M_params)+"M parameters (trainable).")
        print("- Input num channels =", self.num_channels, ", Output num classes=", self.num_classes)
        if example_forward:
            self.example_forward()
        if self.custom_config is not None:
            print("- using custom settings:", self.custom_config)

    def example_forward(self):
        example_input_batch = torch.rand((2, self.num_channels, self.image_resolution, self.image_resolution)) # batch of 2
        example_input_batch = example_input_batch.to(self.device)
        
        example_output_batch = self.model.forward( example_input_batch )
        print("- example input", example_input_batch.shape, "-> example output", example_output_batch.shape)

    # Overrides, for single model simply just pass ---
    def forward(self, *args):
        return self.model.forward(*args)

    def to(self, device):
        self.device = device
        self.model.network.to(device)

    def train(self, b):
        self.model.network.train(b)

    def eval(self):
        self.model.network.eval()

    def parameters(self):
        return self.model.network.parameters()

    def state_dict(self):
        return self.model.network.state_dict()

    def get_network(self):
        return self.model.network
