# Import models
from src.models.CNNten import CNNten, CNNten_multi_task
from src.models.MLPten import MLPten
from src.models.CNNeleven import CNNeleven
from src.models.smallFCN import smallFCN, smallFCN_multi_task, smallFCN_SelfAttention_multi_task, experimentalFCN
from src.models.ViT import ViT1D_multi_task

def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"Total trainable params: {total_params}")

if __name__ == "__main__":
    model = experimentalFCN()
    print_model_parameters(model)

    #model = smallFCN_multi_task()
    #print_model_parameters(model)