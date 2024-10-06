import cv2
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

    # Aplicando o Filtro Mediano
    blurred_median = cv2.medianBlur(gray_image, 5)

    # Plotando os resultados
    plt.figure(figsize=(10, 7))

    # Original em Cinza
    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original (Escala de Cinza)')
    plt.axis('off')

    # Filtro Gaussiano
    plt.subplot(1, 3, 2)
    plt.imshow(blurred_gaussian, cmap='gray')
    plt.title('Filtro Gaussiano')
    plt.axis('off')

    # Filtro Mediano
    plt.subplot(1, 3, 3)
    plt.imshow(blurred_median, cmap='gray')
    plt.title('Filtro Mediano')
    plt.axis('off')

    # Mostra os gráficos
    plt.tight_layout()
    plt.show()