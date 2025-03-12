
# Python 3.10.12
python -m venv venv
source ./venv/bin/activate

# install package from requirements-linux.txt
# or just installed the latest version it should work...

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install tensorboard

pip install stable-baselines3  sb3_contrib Gymnasium

pip install pygame
