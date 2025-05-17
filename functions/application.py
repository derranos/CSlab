import sys
import os
import numpy as np
import cv2
from skimage.color import rgb2lab
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.io import imsave
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap # pip install umap-learn

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QFrame, QMessageBox, QTextEdit, QGridLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Импорт GrabCutDialog
try:
    from grabcut_interactive_tool import GrabCutDialog
except ImportError as e:
    print(f"Критическая ошибка: Не удалось импортировать GrabCutDialog из grabcut_interactive_tool.py: {e}")
    GrabCutDialog = None
except Exception as e_general:
    print(f"Критическая ошибка при импорте GrabCutDialog: {e_general}")
    GrabCutDialog = None


# --- Функция кластеризации (K-Means + HOG/Color) с ИСПРАВЛЕННЫМ UMAP ---
def clustering_main(image_matrices_with_paths, k=5, hog_cell_size=(16, 16), hog_bins=9, hist_bins=64, output_dir="clustered_images_output"):
    if not image_matrices_with_paths:
        raise ValueError("Входной список матриц изображений пуст.")

    image_matrices = [item['matrix'] for item in image_matrices_with_paths]
    original_paths = [item['path'] for item in image_matrices_with_paths]

    hog_features = []
    color_features = []

    for idx, img_orig in enumerate(image_matrices):
        img = img_orig.copy()
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Изображение {original_paths[idx]} должно быть 3D массивом с 3 каналами.")
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.dtype in [np.float32, np.float64]:
                 img = (img * 255).astype(np.uint8)
            elif img.max() <= 255 and img.dtype in [np.float32, np.float64, np.uint16, np.int16, np.int32, np.int64]:
                 img = img.astype(np.uint8)
            else:
                 raise ValueError(f"Изображение {original_paths[idx]} имеет неподдерживаемый тип {img.dtype} или диапазон ({img.min()}-{img.max()}) для конвертации в uint8.")

        img_resized = cv2.resize(img, (256, 256))
        hog = cv2.HOGDescriptor(
            _winSize=(256, 256), _blockSize=(16, 16), _blockStride=(8, 8),
            _cellSize=hog_cell_size, _nbins=hog_bins
        )
        hog_feature = hog.compute(img_resized)
        if hog_feature is None:
            raise ValueError(f"Не удалось извлечь HOG признаки для {original_paths[idx]}.")
        hog_features.append(hog_feature.flatten())

        img_lab = rgb2lab(img_resized)
        hist_l = np.histogram(img_lab[:, :, 0], bins=hist_bins, range=(0, 100))[0]
        hist_a = np.histogram(img_lab[:, :, 1], bins=hist_bins, range=(-128, 127))[0]
        hist_b = np.histogram(img_lab[:, :, 2], bins=hist_bins, range=(-128, 127))[0]
        color_feature = np.concatenate([hist_l, hist_a, hist_b])
        color_features.append(color_feature)

    hog_features = np.array(hog_features)
    color_features = np.array(color_features)

    if hog_features.ndim == 1 and hog_features.size == 0:
        hog_features = np.empty((len(image_matrices), 0))
    if color_features.ndim == 1 and color_features.size == 0:
        color_features = np.empty((len(image_matrices), 0))

    if hog_features.shape[0] == 0 or color_features.shape[0] == 0:
        raise ValueError("Не удалось извлечь HOG или цветовые признаки.")
    if hog_features.shape[0] != color_features.shape[0]:
        raise ValueError(f"Несовпадение HOG ({hog_features.shape[0]}) и цветовых ({color_features.shape[0]}) признаков.")

    combined_features = np.hstack((hog_features, color_features))
    if combined_features.shape[1] == 0: # Если вообще нет признаков
        raise ValueError("Комбинированные признаки пусты.")


    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)

    n_samples = combined_features_scaled.shape[0]
    n_features_original = combined_features_scaled.shape[1]

    if n_samples == 0:
        raise ValueError("Нет признаков для кластеризации после масштабирования.")

    # --- ИСПРАВЛЕННАЯ Логика для UMAP и снижения размерности ---
    min_umap_components = 2
    min_umap_neighbors = 2 

    if n_samples <= min_umap_neighbors or n_samples <= min_umap_components or n_features_original < min_umap_components :
        print(f"Предупреждение: UMAP пропущен (образцов: {n_samples}, признаков: {n_features_original}). Используются масштабированные признаки.")
        reduced_features = combined_features_scaled
    else:
        n_neighbors_for_umap = min(15, n_samples - 1)
        if n_neighbors_for_umap < min_umap_neighbors:
             print(f"Предупреждение: UMAP пропущен, т.к. расчетное n_neighbors ({n_neighbors_for_umap}) < {min_umap_neighbors}. Используются масштабированные признаки.")
             reduced_features = combined_features_scaled
        else:
            max_possible_components = min(n_samples - 1, n_features_original)
            if max_possible_components < min_umap_components:
                print(f"Предупреждение: UMAP пропущен, т.к. макс. n_components ({max_possible_components}) < {min_umap_components}. Используются масштабированные признаки.")
                reduced_features = combined_features_scaled
            else:
                n_components_for_umap = min(50, max_possible_components)
                n_components_for_umap = max(min_umap_components, n_components_for_umap)

                print(f"Info: Запуск UMAP с n_components={n_components_for_umap} и n_neighbors={n_neighbors_for_umap}")
                try:
                    reducer = umap.UMAP(
                        n_components=n_components_for_umap,
                        n_neighbors=n_neighbors_for_umap,
                        random_state=42,
                    )
                    reduced_features = reducer.fit_transform(combined_features_scaled)
                except Exception as e_umap:
                    print(f"Ошибка при выполнении UMAP: {e_umap}. Используются масштабированные признаки.")
                    reduced_features = combined_features_scaled

    # --- Логика для KMeans ---
    n_samples_for_kmeans = reduced_features.shape[0]
    if n_samples_for_kmeans == 0:
        raise ValueError("Нет признаков для KMeans после снижения размерности.")

    actual_k_for_kmeans = min(k, n_samples_for_kmeans) # k - запрошенное пользователем
    if actual_k_for_kmeans == 0 : actual_k_for_kmeans = 1

    if actual_k_for_kmeans != k :
        print(f"Предупреждение: k для KMeans изменен с {k} на {actual_k_for_kmeans} (образцов: {n_samples_for_kmeans}).")
    
    if reduced_features.shape[1] == 0: # Если UMAP вернул 0 признаков
        raise ValueError("После UMAP не осталось признаков для KMeans.")

    kmeans = KMeans(n_clusters=actual_k_for_kmeans, init='k-means++', n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # --- Сохранение результатов ---
    output_paths_by_cluster = [[] for _ in range(actual_k_for_kmeans)]
    for i in range(actual_k_for_kmeans):
        cluster_sub_path = os.path.join(output_dir, f"cluster_{i}")
        if not os.path.exists(cluster_sub_path): os.makedirs(cluster_sub_path)

    saved_image_info = []
    for idx, original_path in enumerate(original_paths):
        label = cluster_labels[idx]
        img_to_save = cv2.imread(original_path)
        if img_to_save is None:
            print(f"Предупреждение: Не удалось перечитать {original_path} для сохранения.")
            continue
        base_filename = os.path.basename(original_path)
        save_path = os.path.join(output_dir, f"cluster_{label}", base_filename)
        cv2.imwrite(save_path, img_to_save)
        output_paths_by_cluster[label].append(save_path)
        saved_image_info.append({'cluster': label, 'path': save_path, 'original_path': original_path})

    result = {
        'cluster_labels': cluster_labels.tolist(),
        'features': reduced_features.tolist(), # Не используется для отображения, но может быть полезно
        'output_paths_by_cluster': output_paths_by_cluster,
        'saved_image_info': sorted(saved_image_info, key=lambda x: (x['cluster'], x['original_path']))
    }
    return result

# --- Функция сегментации Суперпикселями (Felzenszwalb) ---
def superpixels_main(image_matrices_with_paths, scale=80, sigma=0.5, min_size=100, output_dir_superpixels="./superpixels_results"):
    if not image_matrices_with_paths:
        raise ValueError("Входной список матриц изображений пуст.")

    result = {'superpixel_labels_list': [], 'output_paths': []}

    for idx, item in enumerate(image_matrices_with_paths):
        img_rgb_uint8 = item['matrix']
        original_path = item['path']

        if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
            raise ValueError(f"Изображение {original_path} должно быть 3D массивом с 3 каналами.")
        if img_rgb_uint8.dtype != np.uint8:
             raise ValueError(f"Изображение {original_path} имеет неверный тип {img_rgb_uint8.dtype}, ожидался uint8.")

        img_for_felzenszwalb = img_rgb_uint8
        img_float_for_mark = img_rgb_uint8.astype(np.float32) / 255.0

        segments = felzenszwalb(img_for_felzenszwalb, scale=scale, sigma=sigma, min_size=min_size)
        img_with_boundaries = mark_boundaries(img_float_for_mark, segments, color=(1, 0, 0))

        base_filename = os.path.basename(original_path)
        output_filename = f"superpixel_{os.path.splitext(base_filename)[0]}_{idx:03d}.jpg"
        output_path = os.path.join(output_dir_superpixels, output_filename)
        
        imsave(output_path, (img_with_boundaries * 255).astype(np.uint8))
        
        result['superpixel_labels_list'].append(segments.tolist())
        result['output_paths'].append(output_path)
        print(f"Суперпиксели: Изображение {original_path} обработано, сохранено в {output_path}")
    return result

# --- Функция сегментации MeanShift ---
def meanshift_segmentation_main(image_matrices_with_paths, spatial_radius=10, color_radius=20, output_dir_meanshift="./meanshift_app_output"):
    if not image_matrices_with_paths:
        raise ValueError("Входной список матриц изображений пуст.")

    results = {'output_paths': []}

    for idx, item in enumerate(image_matrices_with_paths):
        img_rgb = item['matrix']
        original_path = item['path']

        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"Изображение {original_path} должно быть 3D массивом с 3 каналами.")
        if img_rgb.dtype != np.uint8:
             raise ValueError(f"Изображение {original_path} имеет неверный тип {img_rgb.dtype}, ожидался uint8.")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        segmented_bgr = cv2.pyrMeanShiftFiltering(img_bgr, sp=spatial_radius, sr=color_radius, maxLevel=1,
                                                  termcrit=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1))
        segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

        base_filename = os.path.basename(original_path)
        output_filename = f"meanshift_{os.path.splitext(base_filename)[0]}_{idx:03d}.jpg"
        output_path = os.path.join(output_dir_meanshift, output_filename)

        try:
            imsave(output_path, segmented_rgb)
            results['output_paths'].append(output_path)
            print(f"MeanShift: Изображение {original_path} обработано, сохранено в {output_path}")
        except Exception as e:
            print(f"Ошибка сохранения {output_path} для MeanShift: {e}")
    return results

# --- PyQt6 Worker Thread for processing ---
class ProcessingWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, func, images_with_paths, params, output_dir_base):
        super().__init__()
        self.func = func
        self.images_with_paths = images_with_paths
        self.params = params
        self.output_dir_base = output_dir_base

    def run(self):
        try:
            self.progress.emit(f"Начало обработки: {self.func.__name__}...")
            
            specific_output_subdir = ""
            output_param_name_for_func = ""

            if self.func == clustering_main:
                specific_output_subdir = "clustered_images_output"
                output_param_name_for_func = 'output_dir'
            elif self.func == superpixels_main:
                specific_output_subdir = "superpixels_results"
                output_param_name_for_func = 'output_dir_superpixels'
            elif self.func == meanshift_segmentation_main:
                specific_output_subdir = "meanshift_app_output"
                output_param_name_for_func = 'output_dir_meanshift'
            else:
                raise ValueError(f"Неизвестная функция обработки в воркере: {self.func.__name__}")

            target_output_dir = os.path.join(self.output_dir_base, specific_output_subdir)
            if not os.path.exists(target_output_dir):
                os.makedirs(target_output_dir)
                self.progress.emit(f"Создана папка: {target_output_dir}")
            
            current_processing_params = self.params.copy()
            current_processing_params[output_param_name_for_func] = target_output_dir
            
            result = self.func(self.images_with_paths, **current_processing_params)
            
            self.progress.emit("Обработка завершена.")
            self.finished.emit(result)
        except Exception as e:
            self.progress.emit(f"Ошибка в потоке обработки ({self.func.__name__}): {e}")
            import traceback
            self.progress.emit(traceback.format_exc())
            self.finished.emit(e)


# --- Main Application Window ---
class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработчик Изображений")
        self.setGeometry(100, 100, 1000, 800)
        self.image_matrices_with_paths = []
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        controls_layout = QGridLayout()

        self.input_folder_label = QLabel("Папка с изображениями:")
        self.input_folder_edit = QLineEdit()
        self.input_folder_button = QPushButton("Обзор...")
        self.input_folder_button.clicked.connect(self.browse_input_folder)
        controls_layout.addWidget(self.input_folder_label, 0, 0)
        controls_layout.addWidget(self.input_folder_edit, 0, 1, 1, 2)
        controls_layout.addWidget(self.input_folder_button, 0, 3)

        self.output_folder_label = QLabel("Папка для результатов:")
        self.output_folder_edit = QLineEdit()
        self.output_folder_button = QPushButton("Обзор...")
        self.output_folder_button.clicked.connect(self.browse_output_folder)
        controls_layout.addWidget(self.output_folder_label, 1, 0)
        controls_layout.addWidget(self.output_folder_edit, 1, 1, 1, 2)
        controls_layout.addWidget(self.output_folder_button, 1, 3)

        self.proc_type_label = QLabel("Тип обработки:")
        self.proc_type_combo = QComboBox()
        self.proc_type_combo.addItems([
            "Кластеризация (K-Means + HOG/Color)",
            "Суперпиксели (Felzenszwalb)",
            "MeanShift Сегментация",
            "GrabCut (Интерактивно)"
        ])
        self.proc_type_combo.currentIndexChanged.connect(self.update_param_widgets)
        controls_layout.addWidget(self.proc_type_label, 2, 0)
        controls_layout.addWidget(self.proc_type_combo, 2, 1, 1, 2)
        
        main_layout.addLayout(controls_layout)

        self.params_frame = QFrame()
        self.params_layout = QVBoxLayout(self.params_frame)
        main_layout.addWidget(self.params_frame)

        self.clustering_params_widget = self.create_clustering_params_widget()
        self.superpixels_params_widget = self.create_superpixels_params_widget()
        self.meanshift_params_widget = self.create_meanshift_params_widget()
        self.params_layout.addWidget(self.clustering_params_widget)
        self.params_layout.addWidget(self.superpixels_params_widget)
        self.params_layout.addWidget(self.meanshift_params_widget)

        self.run_button = QPushButton("Запустить обработку")
        self.run_button.clicked.connect(self.run_processing)
        main_layout.addWidget(self.run_button)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True); self.status_log.setFixedHeight(100)
        main_layout.addWidget(QLabel("Лог:")); main_layout.addWidget(self.status_log)

        self.results_scroll_area = QScrollArea()
        self.results_scroll_area.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_scroll_area.setWidget(self.results_widget)
        main_layout.addWidget(QLabel("Результаты:")); main_layout.addWidget(self.results_scroll_area, 1)
        
        self.update_param_widgets()

    def create_clustering_params_widget(self):
        widget = QWidget(); layout = QGridLayout(widget)
        layout.addWidget(QLabel("K (число кластеров):"), 0, 0)
        self.k_spinbox = QSpinBox(); self.k_spinbox.setRange(1, 100); self.k_spinbox.setValue(3)
        layout.addWidget(self.k_spinbox, 0, 1)
        return widget

    def create_superpixels_params_widget(self):
        widget = QWidget(); layout = QGridLayout(widget)
        layout.addWidget(QLabel("Scale:"), 0, 0)
        self.scale_spinbox = QDoubleSpinBox(); self.scale_spinbox.setRange(1.0, 1000.0); self.scale_spinbox.setValue(80.0); self.scale_spinbox.setSingleStep(10.0)
        layout.addWidget(self.scale_spinbox, 0, 1)
        layout.addWidget(QLabel("Sigma:"), 1, 0)
        self.sigma_spinbox = QDoubleSpinBox(); self.sigma_spinbox.setRange(0.01, 10.0); self.sigma_spinbox.setValue(0.5); self.sigma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.sigma_spinbox, 1, 1)
        layout.addWidget(QLabel("Min Size:"), 2, 0)
        self.min_size_spinbox = QSpinBox(); self.min_size_spinbox.setRange(1, 10000); self.min_size_spinbox.setValue(100); self.min_size_spinbox.setSingleStep(10)
        layout.addWidget(self.min_size_spinbox, 2, 1)
        return widget

    def create_meanshift_params_widget(self):
        widget = QWidget(); layout = QGridLayout(widget)
        layout.addWidget(QLabel("Пространственный радиус (sp):"), 0, 0)
        self.ms_spatial_radius_spinbox = QSpinBox(); self.ms_spatial_radius_spinbox.setRange(1, 100); self.ms_spatial_radius_spinbox.setValue(10)
        layout.addWidget(self.ms_spatial_radius_spinbox, 0, 1)
        layout.addWidget(QLabel("Цветовой радиус (sr):"), 1, 0)
        self.ms_color_radius_spinbox = QSpinBox(); self.ms_color_radius_spinbox.setRange(1, 100); self.ms_color_radius_spinbox.setValue(20)
        layout.addWidget(self.ms_color_radius_spinbox, 1, 1)
        return widget

    def update_param_widgets(self):
        current_type = self.proc_type_combo.currentText()
        self.clustering_params_widget.hide(); self.superpixels_params_widget.hide(); self.meanshift_params_widget.hide()
        if "Кластеризация" in current_type: self.clustering_params_widget.show(); self.run_button.setText("Запустить обработку")
        elif "Суперпиксели" in current_type: self.superpixels_params_widget.show(); self.run_button.setText("Запустить обработку")
        elif "MeanShift" in current_type: self.meanshift_params_widget.show(); self.run_button.setText("Запустить обработку")
        elif "GrabCut" in current_type: self.run_button.setText("Открыть GrabCut Инструмент")
        else: self.run_button.setText("Запустить обработку")
        self.log_message(f"Выбран тип: {current_type}")

    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder: self.input_folder_edit.setText(folder); self.load_images_from_folder(folder)

    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для результатов")
        if folder: self.output_folder_edit.setText(folder)

    def log_message(self, message):
        self.status_log.append(message); QApplication.processEvents()

    def load_images_from_folder(self, folder_path):
        self.image_matrices_with_paths = []
        self.clear_results_display()
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        self.log_message(f"Загрузка изображений из: {folder_path}")
        count = 0
        try:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(valid_extensions):
                    file_path = os.path.join(folder_path, filename)
                    img_bgr = cv2.imread(file_path)
                    if img_bgr is None: self.log_message(f"Ошибка: Не удалось загрузить {filename}"); continue
                    img_rgb_uint8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
                    self.image_matrices_with_paths.append({'path': file_path, 'matrix': img_rgb_uint8})
                    count +=1
            self.log_message(f"Загружено {count} изображений.")
            if count == 0: QMessageBox.warning(self, "Нет изображений", f"В '{folder_path}' не найдено ({', '.join(valid_extensions)}).")
        except Exception as e:
            self.log_message(f"Ошибка при загрузке: {e}"); QMessageBox.critical(self, "Ошибка загрузки", f"{e}")

    def run_processing(self):
        input_dir = self.input_folder_edit.text(); output_dir_base = self.output_folder_edit.text()
        proc_type = self.proc_type_combo.currentText()

        if "GrabCut" in proc_type:
            if GrabCutDialog is None: QMessageBox.critical(self, "Ошибка GrabCut", "Компонент GrabCut не загружен."); return
            if not output_dir_base: QMessageBox.warning(self, "Ошибка", "Укажите папку для результатов GrabCut."); return
            if not os.path.exists(output_dir_base):
                try: os.makedirs(output_dir_base)
                except Exception as e: QMessageBox.critical(self, "Ошибка папки", f"{e}"); return
            grabcut_dialog_instance = GrabCutDialog(output_base_dir=output_dir_base, parent=self)
            grabcut_dialog_instance.exec(); self.log_message("GrabCut закрыт.")
            return

        if not input_dir or not os.path.isdir(input_dir): QMessageBox.warning(self, "Ошибка", "Укажите папку с изображениями."); return
        if not output_dir_base: QMessageBox.warning(self, "Ошибка", "Укажите папку для результатов."); return
        if not os.path.exists(output_dir_base):
            try: os.makedirs(output_dir_base); self.log_message(f"Создана папка: {output_dir_base}")
            except Exception as e: QMessageBox.critical(self, "Ошибка папки", f"{e}"); return
        if not self.image_matrices_with_paths: QMessageBox.warning(self, "Нет данных", "Загрузите изображения."); return

        params_for_worker = {}; func_to_run = None
        if "Кластеризация" in proc_type: params_for_worker['k'] = self.k_spinbox.value(); func_to_run = clustering_main
        elif "Суперпиксели" in proc_type:
            params_for_worker['scale']=self.scale_spinbox.value(); params_for_worker['sigma']=self.sigma_spinbox.value(); params_for_worker['min_size']=self.min_size_spinbox.value()
            func_to_run = superpixels_main
        elif "MeanShift" in proc_type:
            params_for_worker['spatial_radius']=self.ms_spatial_radius_spinbox.value(); params_for_worker['color_radius']=self.ms_color_radius_spinbox.value()
            func_to_run = meanshift_segmentation_main
        else: QMessageBox.critical(self, "Ошибка", "Неизвестный тип обработки."); return

        self.run_button.setEnabled(False); self.clear_results_display()
        self.log_message(f"Обработка ({proc_type})...")
        self.thread = ProcessingWorker(func_to_run, self.image_matrices_with_paths, params_for_worker, output_dir_base)
        self.thread.finished.connect(self.on_processing_finished); self.thread.progress.connect(self.log_message)
        self.thread.start()

    def on_processing_finished(self, result_data_or_exception):
        self.run_button.setEnabled(True)
        if isinstance(result_data_or_exception, Exception):
            self.log_message(f"Ошибка обработки: {result_data_or_exception}")
            QMessageBox.critical(self, "Ошибка обработки", f"{result_data_or_exception}")
            return
        self.log_message("Обработка успешно завершена.")
        proc_type = self.proc_type_combo.currentText()
        if "Кластеризация" in proc_type and 'saved_image_info' in result_data_or_exception:
            self.display_clustered_images(result_data_or_exception['saved_image_info'])
        elif ("Суперпиксели" in proc_type or "MeanShift" in proc_type) and 'output_paths' in result_data_or_exception:
            self.log_message(f"Результаты для '{proc_type}':")
            self.display_images(result_data_or_exception['output_paths'])
        else:
            if not isinstance(result_data_or_exception, Exception): self.log_message("Результаты не содержат данных для отображения.")

    def _add_image_to_grid(self, grid_layout, img_path, row, col, tooltip_text=""):
        pixmap = QPixmap(img_path)
        if pixmap.isNull(): self.log_message(f"Не удалось загрузить Pixmap: {img_path}"); return
        img_label = QLabel(); img_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        img_label.setToolTip(tooltip_text if tooltip_text else os.path.basename(img_path))
        filename_label = QLabel(os.path.basename(img_path)); filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter); filename_label.setWordWrap(True)
        cell_widget = QWidget(); cell_layout = QVBoxLayout(cell_widget)
        cell_layout.addWidget(img_label); cell_layout.addWidget(filename_label); cell_layout.setContentsMargins(5,5,5,5)
        grid_layout.addWidget(cell_widget, row, col)

    def display_clustered_images(self, saved_image_info_list):
        self.clear_results_display()
        if not saved_image_info_list: self.log_message("Нет кластеризованных изображений."); return
        self.log_message(f"Отображение {len(saved_image_info_list)} кластеризованных изображений...")
        clusters = {}
        for item in saved_image_info_list:
            cluster_id = item['cluster']
            if cluster_id not in clusters: clusters[cluster_id] = []
            clusters[cluster_id].append(item['path'])
        for cluster_id in sorted(clusters.keys()):
            cluster_title_label = QLabel(f"<b>Кластер {cluster_id}</b> ({len(clusters[cluster_id])} изображений)")
            cluster_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.results_layout.addWidget(cluster_title_label)
            grid_layout_for_cluster = QGridLayout(); row, col, max_cols_per_row = 0, 0, 4
            for img_path in clusters[cluster_id]:
                self._add_image_to_grid(grid_layout_for_cluster, img_path, row, col, f"{os.path.basename(img_path)}\nКластер: {cluster_id}")
                col += 1;
                if col >= max_cols_per_row: col = 0; row += 1
            self.results_layout.addLayout(grid_layout_for_cluster)
            line_sep = QFrame(); line_sep.setFrameShape(QFrame.Shape.HLine); line_sep.setFrameShadow(QFrame.Shadow.Sunken)
            self.results_layout.addWidget(line_sep)

    def display_images(self, image_paths_list):
        self.clear_results_display()
        if not image_paths_list: self.log_message("Нет изображений для отображения."); return
        self.log_message(f"Отображение {len(image_paths_list)} изображений...")
        grid_layout_general = QGridLayout(); row, col, max_cols_per_row = 0, 0, 4
        for img_path in image_paths_list:
            self._add_image_to_grid(grid_layout_general, img_path, row, col)
            col += 1
            if col >= max_cols_per_row: col = 0; row += 1
        self.results_layout.addLayout(grid_layout_general)
        self.results_layout.addStretch(1)

    def clear_results_display(self):
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            elif child.layout():
                while child.layout().count():
                    sub_child = child.layout().takeAt(0)
                    if sub_child.widget(): sub_child.widget().deleteLater()
                child.layout().deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try: import umap
    except ImportError:
        QMessageBox.critical(None, "Ошибка UMAP", "Модуль 'umap-learn' не найден.\nУстановите: pip install umap-learn"); sys.exit(1)
    if GrabCutDialog is None:
        QMessageBox.critical(None, "Ошибка GrabCut", "GrabCut не загружен.\nПроверьте 'grabcut_interactive_tool.py'.")
    main_window = ImageProcessorApp()
    main_window.show()
    sys.exit(app.exec())