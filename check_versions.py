import pkg_resources
import torch
import torchaudio
import platform

print("===== PYTHON VERSION =====")
print(platform.python_version())

print("\n===== TORCH VERSIONS =====")
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except:
    print("torchvision: NOT INSTALLED")

print("\n===== INSTALLED PACKAGES =====")
for pkg in sorted([(d.project_name, d.version) for d in pkg_resources.working_set]):
    print(f"{pkg[0]}=={pkg[1]}")
