import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# limpar a pasta de resultados
if os.path.exists('resultados'):
    for file in os.listdir('resultados'):
        os.remove(os.path.join('resultados', file))

# Caminho para a pasta com as imagens
image_folder = 'dataset/test'

# Listando todas as imagens da pasta
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Verificando se encontrou imagens
if not image_files:
    print("Nenhuma imagem encontrada na pasta.")
else:
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # Carregando a imagem
        image = cv2.imread(image_path)

        if image is None:
            print(f"Erro ao carregar a imagem {image_file}.")
            continue

        # Convertendo para escala de cinza
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
        plate_detected = False

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
                    plate_detected = True  # Placa detectada
                    cv2.drawContours(image_with_contours, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(image_with_contours, "Placa", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Se alguma placa foi detectada, salva a imagem
        if plate_detected:
            output_path = os.path.join('resultados', f'contornos_{image_file}')
            if not os.path.exists('resultados'):
                os.makedirs('resultados')

            plt.figure(figsize=(15, 10))

            # Imagem Original
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Imagem Original')
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

            
            plt.subplot(2, 3, 6)
            plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
            plt.title(f'Contornos e Detecção de Placas')
            plt.axis('off')
            
            # Mostra os gráficos
            plt.tight_layout()
            plt.savefig(output_path)
            
            plt.close()

            print(f"Placa detectada: {image_file} - Resultado salvo em {output_path}")
