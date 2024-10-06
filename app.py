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
    blurred_gaussian = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Equalização de Histograma
    equalized_image = cv2.equalizeHist(blurred_gaussian)

    # Detecção de bordas usando Canny (ajustar parâmetros)
    edges = cv2.Canny(equalized_image, 50, 150)

    # Aplicação de Transformações Morfológicas (fechamento)
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar os contornos na imagem original
    image_with_contours = image.copy()

    # Inicializar variável para salvar o contorno da placa
    plate_contour = None

    for cnt in contours:
        # Aproximar os contornos para verificar se formam um quadrilátero
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Se o contorno aproximado tiver 4 vértices, pode ser a placa (retângulo)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Verificar se a proporção está dentro da faixa típica de uma placa veicular
            if 2 <= aspect_ratio <= 5 and w > 50 and h > 20:
                plate_contour = approx
                cv2.drawContours(image_with_contours, [approx], -1, (0, 255, 0), 3)
                cv2.putText(image_with_contours, "Placa", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Plotando os resultados, mantendo seus gráficos originais
    plt.figure(figsize=(15, 10))

    # Original em Cinza
    plt.subplot(2, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original (Escala de Cinza)')
    plt.axis('off')

    # Filtro Gaussiano
    plt.subplot(2, 3, 2)
    plt.imshow(blurred_gaussian, cmap='gray')
    plt.title('Filtro Gaussiano')
    plt.axis('off')

    # Imagem Equalizada
    plt.subplot(2, 3, 3)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalização de Histograma')
    plt.axis('off')

    # Detecção de Bordas (Canny)
    plt.subplot(2, 3, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Detecção de Bordas (Canny)')
    plt.axis('off')

    # Detecção de Bordas Fechadas (Morfologia)
    plt.subplot(2, 3, 5)
    plt.imshow(closed_edges, cmap='gray')
    plt.title('Bordas Fechadas (Morfologia)')
    plt.axis('off')

    # Mostra os gráficos
    plt.tight_layout()
    plt.show()

    # Exibindo imagem com contornos e possíveis placas detectadas
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Contornos e Detecção de Placas')
    plt.axis('off')
    plt.show()
