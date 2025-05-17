import numpy as np
import cv2
from skimage.color import rgb2lab
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import os
# !!!!!! Работает нормально только для простых изображений, не забыть что пользователь сам вводит !!k!!
def clustering_main(image_matrices, k=5, hog_cell_size=(16, 16), hog_bins=9, hist_bins=64):
    """
    Основная функция для кластеризации изображений на основе признаков HOG и цветовых гистограмм с использованием K-Means.

    Параметры:
    image_matrices (list of numpy arrays): Список матриц изображений (каждая с формой (высота, ширина, 3)).
    k (int): Число кластеров для K-Means. По умолчанию 5.
    hog_cell_size (tuple): Размер ячеек для HOG (ширина, высота). По умолчанию (16, 16).
    hog_bins (int): Число бинов для гистограмм градиентов HOG. По умолчанию 9.
    hist_bins (int): Число бинов для цветовых гистограмм на канал. По умолчанию 64.

    Возвращает:
    dict: Словарь с номерами кластеров для каждого изображения и признаками.
    """
    # Проверка входных данных
    if not image_matrices:
        raise ValueError("Входной список матриц изображений пуст.")

    # Инициализация списков для хранения признаков
    hog_features = []
    color_features = []

    # Обработка каждого изображения
    for idx, img in enumerate(image_matrices):
        # Проверка формата изображения (должно быть 3D с 3 каналами)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Изображение {idx} должно быть 3D массивом с 3 каналами.")

        # Приведение изображения к uint8 (диапазон 0-255), если оно в float
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Изменение размера изображения до 256x256 для консистентности HOG
        img_resized = cv2.resize(img, (256, 256))

        # --- Извлечение HOG признаков ---
        hog = cv2.HOGDescriptor(
            _winSize=(256, 256),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=hog_cell_size,
            _nbins=hog_bins
        )
        hog_feature = hog.compute(img_resized)
        hog_features.append(hog_feature.flatten())

        # --- Извлечение цветовых гистограмм в Lab ---
        img_lab = rgb2lab(img_resized)
        hist_l = np.histogram(img_lab[:, :, 0], bins=hist_bins, range=(0, 100))[0]  # L канал
        hist_a = np.histogram(img_lab[:, :, 1], bins=hist_bins, range=(-128, 127))[0]  # a канал
        hist_b = np.histogram(img_lab[:, :, 2], bins=hist_bins, range=(-128, 127))[0]  # b канал
        color_feature = np.concatenate([hist_l, hist_a, hist_b])
        color_features.append(color_feature)

    # --- Объединение признаков ---
    hog_features = np.array(hog_features)
    color_features = np.array(color_features)
    combined_features = np.hstack((hog_features, color_features))

    # Нормализация комбинированных признаков
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    # --- Снижение размерности с помощью UMAP ---
    n_samples = len(image_matrices)
    n_components = min(50, n_samples - 3)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_features = reducer.fit_transform(combined_features)

    # --- Кластеризация с помощью K-Means ---
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Формирование результата
    result = {
        'cluster_labels': cluster_labels.tolist(),  # Номера кластеров для каждого изображения !!!! НУЖНА ТОЛЬКО ЭТА ШТУКА !!!!
        'features': reduced_features.tolist()       # Признаки после UMAP
    }

    return result

# Пример вызова функции (демонстрация использования)
if __name__ == "__main__":
    # Путь к папке с изображениями
    images_folder = "./functions/images"

    # Проверка существования папки
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Папка '{images_folder}' не найдена.")

    # Список для хранения матриц изображений
    image_matrices = []

    # Чтение всех .jpg файлов из папки
    for filename in os.listdir(images_folder):
        if filename.lower().endswith('.jpg'):
            # Полный путь к файлу
            file_path = os.path.join(images_folder, filename)
            # Загрузка изображения
            img = cv2.imread(file_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {filename}")
                continue
            # Преобразование из BGR (OpenCV) в RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Нормализация значений пикселей в диапазон [0, 1]
            img_normalized = img_rgb.astype(np.float32) / 255.0
            # Добавление в список
            image_matrices.append(img_normalized)

    # Проверка, были ли найдены изображения
    if not image_matrices:
        raise ValueError(f"В папке '{images_folder}' не найдено .jpg изображений.")

    # Вызов основной функции
    result = clustering_main(image_matrices, k=3)

    # Вывод результатов
    print("Номера кластеров:", result['cluster_labels'])
    print("Размер признаков первого изображения:", len(result['features'][0]))
    print("Количество обработанных изображений:", len(image_matrices))