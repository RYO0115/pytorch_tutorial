import torch

print("Python環境: uv venv")
print(f"Pytorch Version: {torch.__version__}")

# MacのGPUが認識されているか
mps_available = torch.backends.mps.is_available()
print(f"Apple Silicon GPU(MPS) is Available? : {mps_available}")

if mps_available:
    # GPU上にテンソルを計算してテスト
    device = torch.device("mps")
    x = torch.rand(3, 3, device=device)
    y = torch.rand(3, 3, device=device)
    z = x @ y

    print("GPU(MPS)は有効です。行列計算が正しく行われました")
    print(x)
    print(y)
    print(z)
else:
    print("GPU(MPS)が認識されていません")
