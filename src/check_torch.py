print("=== check_torch START ===")

import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

try:
    import torch_directml
    dml = torch_directml.device()
    print("DirectML device:", dml)

    x = torch.randn(1, 3, 64, 64).to(dml)
    print("Tensor device:", x.device)
except Exception as e:
    print("DirectML error:", repr(e))

print("=== check_torch END ===")
