import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract

num_img = 0

# limpar a pasta de resultados
if os.path.exists('resultados'):
    for file in os.listdir('resultados'):
        os.remove(os.path.join('resultados', file))

# Caminho para a pasta com as imagens
image_folder = 'dataset/test'

# Listando todas as imagens da pasta
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

def is_rectangle(approx):
    # Calcula os ângulos entre os pontos
    for i in range(4):
        pt1 = approx[i][0] # Ponto atual
        pt2 = approx[(i + 1) % 4][0] # Próximo ponto
        pt3 = approx[(i + 2) % 4][0] # Ponto após o próximo

        v1 = pt2 - pt1 # Vetor 1
        v2 = pt3 - pt2 # Vetor 2

        dot_product = np.dot(v1, v2)    # Produto escalar
        mag_v1 = np.linalg.norm(v1)    # Magnitude do vetor 1
        mag_v2 = np.linalg.norm(v2)   # Magnitude do vetor 2

        angle = np.arccos(dot_product / (mag_v1 * mag_v2)) * 180.0 / np.pi

        if not (80 <= angle <= 100):  # Verifica se o ângulo está próximo de 90°
            return False
    return True

# Verificando se encontrou imagens
if not image_files:
    print("Nenhuma imagem encontrada na pasta.")
else:
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Erro ao carregar a imagem {image_file}.")
            continue

        # escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # threshold
        _, threshold = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)

        #  Gaussiano
        blurred_gaussian = cv2.GaussianBlur(threshold, (3, 3), 0)

        # Equalização
        equalized_image = blurred_gaussian

        # bordas - Canny 
        edges = cv2.Canny(equalized_image, 100, 160)

        # fechamento  bordas
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Desenhar os contornos na imagem original
        image_with_contours = image.copy()

        # Inicializar variável para salvar o contorno da placa
        plate_detected = False

        for cnt in contours:
            perimetro = cv2.arcLength(cnt, True) * 0.02
            approx = cv2.approxPolyDP(cnt, perimetro, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                # Verificar proporção
                if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 20:
                    # Verificar ângulos 
                    if is_rectangle(approx):
                        plate_detected = True  # Placa detectada
                        cv2.drawContours(image_with_contours, [approx], -1, (0, 255, 0), 3)
                        cv2.putText(image_with_contours, "Placa", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Recortar a região da placa
                        plate_region = gray_image[y:y+h, x:x+w]
                        plt.imshow(plate_region, cmap='gray')
                        plt.imsave('plate_region.png', plate_region, cmap='gray')
                        num_img += 1

                        # Usar Tesseract para reconhecer caracteres da placa
                        custom_config = r'--oem 3 --psm 6'
                        plate_text = pytesseract.image_to_string(plate_region, config=custom_config)

                        print(f"Texto reconhecido na placa: {plate_text.strip()}")

                    
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

            print(f"arquivo: {image_file}")

print(f"Total de imagens processadas: {num_img}")