import numpy as np
import cv2
import os
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.io import imsave
# не забыть что пользователь сам вводит scale, sigma, min_size
def superpixels_main(image_matrices, scale=80, sigma=0.5, min_size=100, output_dir="./functions/superpixels"):
    """
    Создание суперпикселей на изображениях с использованием Efficient Graph-Based Image Segmentation.

    Параметры:
    image_matrices (list of numpy arrays): Список матриц изображений (каждая с формой (высота, ширина, 3)).
    scale (float): Параметр масштаба, влияет на размер суперпикселей (больше scale -> крупнее сегменты).
    sigma (float): Сигма для гауссова сглаживания перед сегментацией.
    min_size (int): Минимальный размер суперпикселя (в пикселях).
    output_dir (str): Папка для сохранения изображений с границами суперпикселей.

    Возвращает:
    dict: Словарь с метками суперпикселей для каждого изображения и путями к сохранённым изображениям.
    """
    # Проверка входных данных
    if not image_matrices:
        raise ValueError("Входной список матриц изображений пуст.")

    # Создание выходной папки, если не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Результаты
    result = {
        'superpixel_labels': [],  # Список массивов меток суперпикселей
        'output_paths': []        # Пути к сохранённым изображениям
    }

    # Обработка каждого изображения
    for idx, img in enumerate(image_matrices):
        # Проверка формата изображения
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Изображение {idx} должно быть 3D массивом с 3 каналами.")

        # Приведение к диапазону [0, 1], если не в нём
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        # Применение Efficient Graph-Based Image Segmentation
        segments = felzenszwalb(
            img,
            scale=scale,
            sigma=sigma,
            min_size=min_size
        )

        # Отметка границ суперпикселей на изображении
        img_with_boundaries = mark_boundaries(img, segments, color=(1, 0, 0))  # Красные границы

        # Сохранение результата
        output_path = os.path.join(output_dir, f"superpixel_{idx:03d}.jpg")
        imsave(output_path, (img_with_boundaries * 255).astype(np.uint8))

        # Сохранение меток и пути
        result['superpixel_labels'].append(segments.tolist())
        result['output_paths'].append(output_path)

        # Диагностика
        num_segments = len(np.unique(segments))
        print(f"Изображение {idx}: {num_segments} суперпикселей, сохранено в {output_path}")

    return result

if __name__ == "__main__":
    # Путь к папке с изображениями
    images_folder = "images"

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
    result = superpixels_main(
        image_matrices,
        scale=70,
        sigma=0.5,
        min_size=100,
        output_dir="superpixels"
    )
    #Поле result['superpixel_labels'] содержит списки массивов меток,
    #где каждый массив имеет форму (высота, ширина) и указывает номер суперпикселя для каждого пикселя.
    # Вывод результатов
    print("Пути к сохранённым изображениям:", result['output_paths'])
    print("Количество обработанных изображений:", len(image_matrices))
    print("Размер меток суперпикселей для первого изображения:", np.array(result['superpixel_labels'][0]).shape)