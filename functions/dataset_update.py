import os
import cv2
import numpy as np

def data_update_main(image_matrices, rotations=(90, 180, 270), do_flip=True, noise_variance=0.01, output_dir="./augmented_data"):
    """
    Увеличение датасета изображений с помощью аугментации:
      - Повороты
      - Отзеркаливание
      - Добавление гауссова шума

    Параметры:
    image_matrices (list of numpy arrays): Список RGB-изображений в диапазоне [0,1].
    rotations (tuple of ints): Углы поворота в градусах.
    do_flip (bool): Добавлять горизонтальное и вертикальное отражение.
    noise_variance (float): Дисперсия гауссова шума.
    output_dir (str): Папка для сохранения аугментированных изображений.

    Возвращает:
    dict: Словарь с ключами:
      'output_paths' – список путей к сохранённым изображениям.
    """
    if not image_matrices:
        raise ValueError("Входной список матриц изображений пуст.")

    os.makedirs(output_dir, exist_ok=True)
    result = {'output_paths': []}

    for idx, img in enumerate(image_matrices):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Изображение {idx} должно быть RGB (3 канала).")
        # Приведение к [0,255]
        img_uint8 = (img * 255).astype(np.uint8)

        # Оригинал
        orig_path = os.path.join(output_dir, f"img_{idx:03d}_orig.jpg")
        cv2.imwrite(orig_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        result['output_paths'].append(orig_path)

        # Повороты
        h, w = img_uint8.shape[:2]
        center = (w // 2, h // 2)
        for angle in rotations:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_uint8, M, (w, h))
            path = os.path.join(output_dir, f"img_{idx:03d}_rot{angle}.jpg")
            cv2.imwrite(path, cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
            result['output_paths'].append(path)

        # Отзеркаливание
        if do_flip:
            # Горизонтальное
            flip_h = cv2.flip(img_uint8, 1)
            path_h = os.path.join(output_dir, f"img_{idx:03d}_fliph.jpg")
            cv2.imwrite(path_h, cv2.cvtColor(flip_h, cv2.COLOR_RGB2BGR))
            result['output_paths'].append(path_h)
            # Вертикальное
            flip_v = cv2.flip(img_uint8, 0)
            path_v = os.path.join(output_dir, f"img_{idx:03d}_flipv.jpg")
            cv2.imwrite(path_v, cv2.cvtColor(flip_v, cv2.COLOR_RGB2BGR))
            result['output_paths'].append(path_v)

        # Гауссов шум
        noise = np.random.normal(0, np.sqrt(noise_variance), img_uint8.shape)
        noisy = img_uint8.astype(np.float32) + noise * 255
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        path_n = os.path.join(output_dir, f"img_{idx:03d}_noise.jpg")
        cv2.imwrite(path_n, cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))
        result['output_paths'].append(path_n)

    print(f"Аугментированные изображения сохранены в папке: {output_dir}")
    return result

if __name__ == "__main__":
    # Путь к папке с изображениями
    images_folder = "./functions/images"
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Папка '{images_folder}' не найдена.")

    image_matrices = []
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(images_folder, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"Не удалось загрузить изображение: {filename}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image_matrices.append(img_rgb)

    if not image_matrices:
        raise ValueError("В папке не найдены поддерживаемые изображения.")

    result = data_update_main(
        image_matrices,
        rotations=(90, 180, 270),
        do_flip=True,
        noise_variance=0.01,
        output_dir="./functions/augmented_images"
    )
    print("Пути к всем аугментированным изображениям:", result['output_paths'])
