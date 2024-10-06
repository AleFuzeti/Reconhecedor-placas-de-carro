import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminho da imagem
image_path = 'dataset/train/BAG-7751_jpg.rf.d460abea0e44d583c1cb8eefc44869bc.jpg'

# Importando a imagem do diretório
image = cv2.imread(image_path)

# Verificando se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    # Convertendo para escala de cinza para facilitar a aplicação dos filtros
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicando o Filtro Gaussiano
    blurred_gaussian = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Equalização de Histograma
    equalized_image = cv2.equalizeHist(blurred_gaussian)

    # Detecção de bordas usando Canny
    edges = cv2.Canny(equalized_image, 50, 200)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar os contornos na imagem original
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Desenha em verde

    # Aplicando a Transformada de Hough para detectar linhas
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=20)

    # Criar uma cópia da imagem original para desenhar as linhas de Hough
    hough_image = image.copy()

    # Verifica se foram encontradas linhas
    if lines is not None:
        # Desenhar as linhas detectadas na imagem
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Desenha as linhas em amarelo

    # Plotando os resultados
    plt.figure(figsize=(15, 10))

    # Original em Cinza
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original (Escala de Cinza)')
    plt.axis('off')

    # Filtro Gaussiano
    plt.subplot(2, 2, 2)
    plt.imshow(blurred_gaussian, cmap='gray')
    plt.title('Filtro Gaussiano')
    plt.axis('off')

    # Imagem Equalizada
    plt.subplot(2, 2, 3)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalização de Histograma')
    plt.axis('off')

    # Detecção de Bordas (Canny)
    plt.subplot(2, 2, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Detecção de Bordas (Canny)')
    plt.axis('off')

    # Mostra os gráficos
    plt.tight_layout()
    plt.show()

    # Exibindo imagem com contornos
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Contornos Detectados')
    plt.axis('off')
    plt.show()

    # Exibindo imagem com Transformada de Hough
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(hough_image, cv2.COLOR_BGR2RGB))
    plt.title('Linhas Detectadas (Transformada de Hough)')
    plt.axis('off')
    plt.show()
