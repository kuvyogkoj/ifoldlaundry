import torch
import transformers

from PIL import Image
from transformers import Sam3Model, Sam3Processor

def main() -> None:
    print("Pytorch Version: ", torch.__version__)
    print("Transformer Version: ", transformers.__version__)
    print("CUDA availiable: ", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU: ", torch.cuda.get_device_name(0))

    print("Pillow imported")
    print("SAM3Model and SAM3Processor imported")

if __name__ == "__main__":
    main()