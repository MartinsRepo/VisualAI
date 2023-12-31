![File not found](intro.png)
# VisualAI - Textual Facedetection
**Python Version used:** 3.11.4
## A) Installation
### 1. Python Environment
A virtual environment is a Python environment such that the Python interpreter, libraries and scripts installed into it are isolated from those installed in other virtual environments.
#### Windows:
**Prerequisites:** Installation of the latest Python version from here:
 - https://www.python.org/downloads/windows/
and and the Pip installer from here:
 - https://phoenixnap.com/kb/install-pip-windows
Then Install **pyenv for Windows**
>pip install pyenv-win
#### Linux:
- Install and update dependencies:
> sudo apt update -y

> sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

- Prerequisites: Installation of the latest Pyenv version from here:
> curl https://pyenv.run | bash

Update .bashrc:
> echo 'export PYENV_ROOT="$HOME/.pyenv"'  >> ~/.bashrc
> echo 'export PATH="$PYENV_ROOT/bin:$PATH"'  >> ~/.bashrc
> echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi'  >> ~/.bashrc

**Setting up the Virtual Environment:**
(1) Check pyenv version
> pyenv versions

(2) Install Python
> pyenv install 3.11.4

(3) Set this version local
> pyenv local 3.11.4

(4) Install **virtualvenv**
> python -m pip install virtualenv

(5) Create your environment
> pyenv virtualenv 3.11.4 visionai

> pyenv activate visionai

### 2. VisualAI installation
> mkdir ~/workspace/

> pip install -r requirements.txt

> cd ~/workspace/VisualAI

### 3. Running VisualAI
> streamlit run facedetection.py

### 2. Python Packages
**Prerequisites:** Compiling and installing the llama.cpp package, under windows, unfortunaltly we must install (if not done already) **Visual Studio** (here: 2022 Community Edition):

https://visualstudio.microsoft.com/de/downloads/

**Installing Dependencies:**
After activating the environment:
>pip install -r requirements.txt


> Written with [StackEdit](https://stackedit.io/).
