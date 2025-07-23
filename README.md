# 🚗 Reconhecedor de Placas de Carro

Projeto de visão computacional para **detecção e reconhecimento de placas de veículos** utilizando OpenCV e OCR com Tesseract.

## 📷 Funcionalidades

- Detecção de placas em imagens usando técnicas de processamento de imagem
- Reconhecimento de caracteres utilizando OCR (Tesseract)
- Visualização e extração de texto das placas

## 🛠️ Dependências

Certifique-se de ter o Python instalado. Em seguida, instale as bibliotecas necessárias:

### 📦 Instalação das dependências

```bash
pip install opencv-python numpy matplotlib pytesseract

### 🧱 Dependências do sistema (Linux)

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo apt install libgl1-mesa-glx
```

> Obs: `libgl1-mesa-glx` é necessário para evitar erros ao usar o OpenCV com interfaces gráficas.

### 🔍 Verifique a instalação do Tesseract

```bash
tesseract --version
```

Se esse comando funcionar, o OCR está pronto para uso.

## 📁 Estrutura do Projeto

- `main.py`: código principal que realiza a detecção e o reconhecimento
- `placas/`: imagens de entrada com placas de veículos
- `resultados/`: saída com texto extraído e imagens anotadas

## 🧪 Como executar

1. Coloque as imagens com placas na pasta `placas/`
2. Execute o script:

```bash
python main.py
```

> Este projeto tem fins educacionais e de experimentação com visão computacional.
```
