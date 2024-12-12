import torch

def check_pytorch_gpu():
    try:
        if torch.cuda.is_available():
            print(f"PyTorch can access {torch.cuda.device_count()} GPU(s).")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch cannot access any GPUs.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_pytorch_gpu()