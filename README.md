## Pre-processing Pipeline and Imaging Equipment Impact the Performance of CNN for Breast Cancer Detection in Mammograms: Evidence from CBIS-DDSM and VinDr-Mammo
------------------------------------------------
## 1. Files

code/pt/learners

* local_mammo_learner.py - arquivo principal que possui funções específicas para o treinamento do modelo, carregamento de pesos, salvar pesos de modelo treinado, entre outros. Esta é a classe que será executada para a realização dos testes de pre-processamentos.

code/pt/utils
* birads_categories.json - arquivo que representa o agrupamento dos bi-rads utilizado na redução de classes.
* img_utils.py - arquivo com funções auxiliares para edição das imagens utilizadas nos pre-processamentos.
preprocess_dicom - arquivo responsável por carregar o arquivo DICOM, extrair os bytes das imagens e depois realizar os pre-processamentos.
* preprocess_json.py - arquivo com funções auxiliares responsáveis pela leitura do arquivo JSON do dataset utilizado nos testes.


