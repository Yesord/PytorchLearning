import torch
import time

def check_cuda(): # 检查CUDA是否可用
    if torch.cuda.is_available(): # torch.cuda.is_available()函数用于检查CUDA是否可用
        print("CUDA is available") # 如果CUDA可用，则打印CUDA is available
    else:
        print("CUDA is not available") # 如果CUDA不可用，则打印CUDA is not available

def check_cudnn(): # 检查cuDNN是否可用
    if torch.backends.cudnn.is_available(): # torch.backends.cudnn.is_available()函数用于检查cuDNN是否可用
        print("cuDNN is available") # 如果cuDNN可用，则打印cuDNN is available
    else:
        print("cuDNN is not available") # 如果cuDNN不可用，则打印cuDNN is not available

def check_torch_cuda_version(): # 检查PyTorch的CUDA版本
    print("PyTorch Version: ", torch.__version__) # 打印PyTorch版本
    print("CUDA Version: ", torch.version.cuda) # 打印CUDA版本 
    print("cuDNN Version: ", torch.backends.cudnn.version()) # 打印cuDNN版本
    print("Device Name: ", torch.cuda.get_device_name(0)) # 打印设备名称
    print("Device Count: ", torch.cuda.device_count()) # 打印设备数量


print("Pytorch_Cuda_Verify.py is running...")
time.sleep(1)