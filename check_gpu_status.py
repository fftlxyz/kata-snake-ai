import torch

if torch.accelerator.is_available():
    print("accelerator found:", torch.accelerator.current_accelerator())
    x = torch.ones(1, device=torch.accelerator.current_accelerator())
    print(x)
else:
    print("no accelerator device not found.")
