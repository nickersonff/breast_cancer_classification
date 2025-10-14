## Criação do projeto no super computador (NPAD)

Tutorial do NPAD
[Tutorial](https://github.com/NPAD-UFRN/Tutorials).

------------------------------------------------
## 1. Criar um ambiente conda

Primeiro é necessário criar um ambiente conda com todas as bibliotecas compatíveis com a versão do NVFlare e projeto usado como base.
ex: conda create --name bio-images python=3.8.12 numpy=1.23.4 pandas

### 1.1 Instalar algumas bibliotecas (com base no Dockerfile)
pip install tensorboard==2.6.0 scikit-learn torchvision==0.11.1
pip install monai==0.8.1
pip install nvflare==2.0.16
pip install protobuf==3.20.* --force-reinstall
pip install setuptools==58.2.0
pip uninstall google-auth

### 1.2 Instalar bibliotecas para os pré-processamentos
pip install pydicom
pip install opencv-python-headless

### 1.3 Instalar as bibliotecas com a mesma versão do container
pip install -r requirements.txt

### 1.4 Criação do ambiente do FL
sed -i "s|{SERVER_FQDN}|localhost|g" fl_project.yml
python3 -m nvflare.lighter.provision -p fl_project.yml
cp -r workspace/fl_project/prod_00 fl_workspace
mv fl_workspace/${server_fqdn} fl_workspace/server

### 1.5 Tentar executar
sbatch run_sc_all_fl.sh
