# meanshift_segmentation.py
import numpy as np
import cv2
import os
from skimage.io import imsave # Using skimage.io for consistency if preferred, or cv2.imwrite

def meanshift_segmentation_main(image_matrices_with_paths, spatial_radius=10, color_radius=20, output_dir_meanshift="./meanshift_output"):
    """
    Применение MeanShift сегментации (через cv2.pyrMeanShiftFiltering) к изображениям.

    Параметры:
    image_matrices_with_paths (list of dicts): Список {'path': str, 'matrix': np.array}.
                                                Матрицы должны быть в формате RGB, uint8.
    spatial_radius (int): Пространственный радиус окна (sp).
    color_radius (int): Цветовой радиус окна (sr).
    output_dir_meanshift (str): Папка для сохранения сегментированных изображений.

    Возвращает:
    dict: Словарь с путями к сохранённым изображениям {'output_paths': []}.
    """
    if not image_matrices_with_paths:
        raise ValueError("Входной список матриц изображений пуст.")

    if not os.path.exists(output_dir_meanshift):
        os.makedirs(output_dir_meanshift)
        print(f"Создана папка: {output_dir_meanshift}")

    results = {'output_paths': []}

    for idx, item in enumerate(image_matrices_with_paths):
        img_rgb_orig = item['matrix'] # Ожидается RGB
        original_path = item['path']

        if img_rgb_orig.ndim != 3 or img_rgb_orig.shape[2] != 3:
            raise ValueError(f"Изображение {original_path} должно быть 3D массивом с 3 каналами.")

        # pyrMeanShiftFiltering ожидает uint8. Конвертируем, если нужно.
        if img_rgb_orig.dtype != np.uint8:
            if img_rgb_orig.max() <= 1.0 and img_rgb_orig.min() >= 0.0 and img_rgb_orig.dtype in [np.float32, np.float64]:
                print(f"Info: Изображение {original_path} типа {img_rgb_orig.dtype}, конвертируется в uint8.")
                img_rgb = (img_rgb_orig * 255).astype(np.uint8)
            else:
                raise ValueError(f"Изображение {original_path} имеет неподдерживаемый тип {img_rgb_orig.dtype} или диапазон для конвертации в uint8.")
        else:
            img_rgb = img_rgb_orig.copy()


        # cv2.pyrMeanShiftFiltering ожидает BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Применение MeanShift фильтрации
        # maxLevel=1 и termcrit можно настроить для более точных/медленных результатов
        segmented_bgr = cv2.pyrMeanShiftFiltering(img_bgr, sp=spatial_radius, sr=color_radius, maxLevel=1, 
                                                  termcrit=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
        
        # Конвертируем результат обратно в RGB для сохранения и отображения
        segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

        base_filename = os.path.basename(original_path)
        output_filename = f"meanshift_{os.path.splitext(base_filename)[0]}_{idx:03d}.jpg"
        output_path = os.path.join(output_dir_meanshift, output_filename)

        try:
            imsave(output_path, segmented_rgb) # segmented_rgb is uint8
            # Или используйте cv2.imwrite:
            # cv2.imwrite(output_path, cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))
            results['output_paths'].append(output_path)
            print(f"MeanShift: Изображение {original_path} обработано, сохранено в {output_path}")
        except Exception as e:
            print(f"Ошибка сохранения {output_path}: {e}")
            # Продолжаем обработку других изображений

    return results

if __name__ == '__main__':
    # Пример использования (для тестирования этого файла отдельно)
    # Создайте папку 'test_images_for_meanshift' с несколькими .jpg или .png изображениями.
    test_img_folder = "test_images_for_meanshift"
    if not os.path.exists(test_img_folder):
        os.makedirs(test_img_folder)
        # Пример создания простого изображения:
        img_data = np.zeros((100,100,3), dtype=np.uint8)
        img_data[:50,:50] = [255,0,0] #R
        img_data[50:,:50] = [0,255,0] #G
        img_data[:50,50:] = [0,0,255] #B
        img_data[50:,50:] = [255,255,0] #Y
        cv2.imwrite(os.path.join(test_img_folder, "colors.png"), cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
        print(f"Создана тестовая папка {test_img_folder} с изображением colors.png. Добавьте еще изображений для теста.")

    image_matrices_with_paths_test = []
    for filename in os.listdir(test_img_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(test_img_folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_matrices_with_paths_test.append({'path': path, 'matrix': img_rgb})

    if not image_matrices_with_paths_test:
        print(f"Не найдены изображения в '{test_img_folder}'")
    else:
        print(f"Найдено {len(image_matrices_with_paths_test)} изображений для теста.")
        try:
            results = meanshift_segmentation_main(
                image_matrices_with_paths_test,
                spatial_radius=20, # увеличено для более заметного эффекта
                color_radius=40,   # увеличено
                output_dir_meanshift="meanshift_test_output_standalone"
            )
            print("Результаты MeanShift (тест):", results['output_paths'])
        except Exception as e:
            print(f"Ошибка при тестировании MeanShift: {e}")