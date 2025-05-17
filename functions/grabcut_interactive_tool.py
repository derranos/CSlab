# grabcut_interactive_tool.py
import sys
import cv2
import numpy as np
import os
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QRadioButton, QButtonGroup, QFrame, QSpinBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPolygonF
from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QPointF

class GrabCutCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.display_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.drawing_mode = 'rect'
        self.rect_start_point = None
        self.current_rect_draw = QRect()
        self.final_rect = None

        self.current_stroke_draw = []
        self.fg_strokes_list = []
        self.bg_strokes_list = []

        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #333;")

    def set_image(self, cv_image_rgb_uint8):
        if cv_image_rgb_uint8 is None:
            self.original_pixmap = None
            self.clear_all_overlays()
            self.update_display_pixmap()
            return

        if cv_image_rgb_uint8.dtype != np.uint8:
            if cv_image_rgb_uint8.max() <= 255 and cv_image_rgb_uint8.min() >=0:
                 cv_image_rgb_uint8 = cv_image_rgb_uint8.astype(np.uint8)
            else:
                 pass # Оставляем как есть

        h, w, ch = cv_image_rgb_uint8.shape
        bytes_per_line = ch * w
        q_img = QImage(cv_image_rgb_uint8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.original_pixmap = QPixmap.fromImage(q_img)
        self.clear_all_overlays()
        self.update_display_pixmap()


    def clear_all_overlays(self):
        self.final_rect = None
        self.current_rect_draw = QRect()
        self.rect_start_point = None
        self.fg_strokes_list = []
        self.bg_strokes_list = []
        self.current_stroke_draw = []

    def _transform_point_to_image(self, widget_point: QPoint) -> QPoint:
        if not self.original_pixmap or self.scale_factor == 0:
            return widget_point 
        
        relative_x = widget_point.x() - self.offset_x
        relative_y = widget_point.y() - self.offset_y
        img_x = relative_x / self.scale_factor
        img_y = relative_y / self.scale_factor
        
        img_x_clamped = max(0, min(img_x, self.original_pixmap.width() - 1))
        img_y_clamped = max(0, min(img_y, self.original_pixmap.height() - 1))
        return QPoint(int(img_x_clamped), int(img_y_clamped))

    def _transform_rect_to_widget(self, image_rect: QRect) -> QRect:
        if not self.original_pixmap or self.scale_factor == 0:
            return image_rect
        
        x = image_rect.x() * self.scale_factor + self.offset_x
        y = image_rect.y() * self.scale_factor + self.offset_y
        w = image_rect.width() * self.scale_factor
        h = image_rect.height() * self.scale_factor
        return QRect(int(x), int(y), int(w), int(h))

    def _transform_stroke_to_widget(self, image_stroke: list[QPoint]) -> list[QPointF]:
        if not self.original_pixmap or self.scale_factor == 0:
            return [QPointF(p.x(), p.y()) for p in image_stroke]

        return [QPointF(p.x() * self.scale_factor + self.offset_x, 
                        p.y() * self.scale_factor + self.offset_y) for p in image_stroke]


    def mousePressEvent(self, event: QPoint): # event.pos() возвращает QPoint
        if not self.original_pixmap or event.button() != Qt.MouseButton.LeftButton:
            return
        
        img_coord_pt = self._transform_point_to_image(event.pos())

        if self.drawing_mode == 'rect':
            self.rect_start_point = img_coord_pt
            self.current_rect_draw = QRect(self.rect_start_point, QSize(1,1))
        elif self.drawing_mode in ['fg_stroke', 'bg_stroke']:
            self.current_stroke_draw = [img_coord_pt]
        self.update_display_pixmap()

    def mouseMoveEvent(self, event: QPoint): # event.pos() возвращает QPoint
        if not self.original_pixmap or not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        img_coord_pt = self._transform_point_to_image(event.pos())

        if self.drawing_mode == 'rect' and self.rect_start_point:
            self.current_rect_draw = QRect(self.rect_start_point, img_coord_pt).normalized()
        elif self.drawing_mode in ['fg_stroke', 'bg_stroke'] and self.current_stroke_draw:
            if not self.current_stroke_draw or self.current_stroke_draw[-1] != img_coord_pt:
                 self.current_stroke_draw.append(img_coord_pt)
        self.update_display_pixmap()

    def mouseReleaseEvent(self, event: QPoint): # event.pos() возвращает QPoint
        if not self.original_pixmap or event.button() != Qt.MouseButton.LeftButton:
            return

        # img_coord_pt = self._transform_point_to_image(event.pos()) # Не используется здесь, т.к. current_stroke_draw уже заполнен

        if self.drawing_mode == 'rect' and self.rect_start_point:
            # self.final_rect устанавливается в mouseMove, здесь только сбрасываем
            if self.current_rect_draw.isValid() and (self.current_rect_draw.width() >= 5 and self.current_rect_draw.height() >= 5):
                self.final_rect = self.current_rect_draw # Фиксируем последний нарисованный прямоугольник
            else:
                self.final_rect = None # Считаем невалидным, если слишком маленький
            self.rect_start_point = None
            self.current_rect_draw = QRect()
        elif self.drawing_mode == 'fg_stroke' and self.current_stroke_draw:
            if len(self.current_stroke_draw) > 1:
                self.fg_strokes_list.append(list(self.current_stroke_draw))
            self.current_stroke_draw = []
        elif self.drawing_mode == 'bg_stroke' and self.current_stroke_draw:
            if len(self.current_stroke_draw) > 1:
                self.bg_strokes_list.append(list(self.current_stroke_draw))
            self.current_stroke_draw = []
        self.update_display_pixmap()

    def update_display_pixmap(self):
        if not self.original_pixmap:
            super().setPixmap(QPixmap())
            return

        scaled_size = self.original_pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.display_pixmap = self.original_pixmap.scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        if self.original_pixmap.width() > 0 and self.original_pixmap.height() > 0 :
             self.scale_factor = min(
                 self.display_pixmap.width() / self.original_pixmap.width() if self.original_pixmap.width() > 0 else 1.0,
                 self.display_pixmap.height() / self.original_pixmap.height() if self.original_pixmap.height() > 0 else 1.0
            )
        else:
             self.scale_factor = 1.0

        self.offset_x = (self.width() - self.display_pixmap.width()) / 2
        self.offset_y = (self.height() - self.display_pixmap.height()) / 2

        painter = QPainter(self.display_pixmap)
        
        if self.final_rect and self.final_rect.isValid():
            pen = QPen(QColor("red"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self._transform_rect_to_widget(self.final_rect))

        if self.drawing_mode == 'rect' and self.current_rect_draw.isValid() and not self.current_rect_draw.isEmpty():
            pen = QPen(QColor("magenta"), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._transform_rect_to_widget(self.current_rect_draw))
        
        pen_fg = QPen(QColor(0, 255, 0, 180), 3, Qt.PenStyle.SolidLine)
        pen_fg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_fg)
        for stroke in self.fg_strokes_list:
            if len(stroke) > 1:
                painter.drawPolyline(QPolygonF(self._transform_stroke_to_widget(stroke)))
        if self.drawing_mode == 'fg_stroke' and self.current_stroke_draw and len(self.current_stroke_draw) > 1:
            painter.drawPolyline(QPolygonF(self._transform_stroke_to_widget(self.current_stroke_draw)))

        pen_bg = QPen(QColor(0, 0, 255, 180), 3, Qt.PenStyle.SolidLine)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bg)
        for stroke in self.bg_strokes_list:
            if len(stroke) > 1:
                painter.drawPolyline(QPolygonF(self._transform_stroke_to_widget(stroke)))
        if self.drawing_mode == 'bg_stroke' and self.current_stroke_draw and len(self.current_stroke_draw) > 1:
            painter.drawPolyline(QPolygonF(self._transform_stroke_to_widget(self.current_stroke_draw)))
        
        painter.end()
        super().setPixmap(self.display_pixmap)

    def get_grabcut_params(self):
        rect_cv = None
        if self.final_rect and self.final_rect.isValid():
            r = self.final_rect
            rect_cv = (r.x(), r.y(), r.width(), r.height())
        
        return rect_cv, list(self.fg_strokes_list), list(self.bg_strokes_list)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_pixmap()


class GrabCutDialog(QDialog):
    def __init__(self, output_base_dir, parent=None):
        super().__init__(parent)
        self.output_base_dir = output_base_dir
        self.setWindowTitle("Инструмент GrabCut Сегментации")
        self.setMinimumSize(800, 700)

        self.cv_image_rgb_orig = None
        self.cv_image_rgb_display = None
        
        self.cv_mask = None
        self.bgd_model = np.zeros((1, 65), dtype=np.float64)
        self.fgd_model = np.zeros((1, 65), dtype=np.float64)
        self.grabcut_iter_count_default = 5
        
        self.is_segmented = False
        self.original_filename_for_saving = "grabcut_output.png"

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        top_controls_layout = QHBoxLayout()
        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image_dialog)
        top_controls_layout.addWidget(self.load_button)
        
        self.iter_label = QLabel("Итераций GrabCut:")
        top_controls_layout.addWidget(self.iter_label)
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setRange(1, 20)
        self.iter_spinbox.setValue(self.grabcut_iter_count_default)
        top_controls_layout.addWidget(self.iter_spinbox)

        self.reset_button = QPushButton("Сбросить разметку")
        self.reset_button.clicked.connect(lambda: self.reset_overlays_and_segmentation(full_reset_models=False))
        top_controls_layout.addWidget(self.reset_button)
        main_layout.addLayout(top_controls_layout)

        self.canvas = GrabCutCanvas(self)

        draw_mode_frame = QFrame()
        draw_mode_frame.setFrameShape(QFrame.Shape.StyledPanel)
        draw_mode_layout = QHBoxLayout(draw_mode_frame)
        self.mode_group = QButtonGroup(self)
        
        modes = [("ROI (прямоугольник)", 'rect'), ("Передний план (FG)", 'fg_stroke'), ("Задний план (BG)", 'bg_stroke')]
        initial_mode_is_set_flag = False
        for i, (text, mode_id) in enumerate(modes):
            radio = QRadioButton(text)
            setattr(radio, 'mode_id', mode_id)
            self.mode_group.addButton(radio)
            draw_mode_layout.addWidget(radio)
            if i == 0: 
                radio.setChecked(True)
                initial_mode_is_set_flag = True 
            radio.toggled.connect(self.on_draw_mode_changed)
        
        main_layout.addWidget(draw_mode_frame)
        main_layout.addWidget(self.canvas, 1)

        action_layout = QHBoxLayout()
        self.segment_button = QPushButton("Выполнить GrabCut")
        self.segment_button.clicked.connect(self.run_segmentation_process)
        action_layout.addWidget(self.segment_button)

        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_segmentation_result)
        self.save_button.setEnabled(False)
        action_layout.addWidget(self.save_button)
        
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        action_layout.addWidget(self.close_button)

        main_layout.addLayout(action_layout)
        self.setLayout(main_layout)

        if initial_mode_is_set_flag:
            checked_button = self.mode_group.checkedButton()
            if checked_button and hasattr(self, 'canvas') and self.canvas:
                self.canvas.drawing_mode = getattr(checked_button, 'mode_id', 'rect')


    def on_draw_mode_changed(self):
        radio_button = self.sender()
        if radio_button and radio_button.isChecked():
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.drawing_mode = getattr(radio_button, 'mode_id', 'rect')


    def load_image_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if filepath:
            try:
                temp_img_bgr = cv2.imread(filepath)
                if temp_img_bgr is None:
                    raise Exception(f"Не удалось прочитать файл изображения: {filepath}")
                
                self.cv_image_rgb_orig = cv2.cvtColor(temp_img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
                self.cv_image_rgb_display = self.cv_image_rgb_orig.copy()
                
                self.canvas.set_image(self.cv_image_rgb_display)
                self.reset_overlays_and_segmentation(full_reset_models=True)
                
                self.original_filename_for_saving = os.path.basename(filepath)
                self.save_button.setEnabled(False)
                self.is_segmented = False
            except Exception as e:
                QMessageBox.warning(self, "Ошибка загрузки", f"Не удалось загрузить изображение: {e}")
                self.cv_image_rgb_orig = None
                self.cv_image_rgb_display = None
                if hasattr(self, 'canvas') and self.canvas: self.canvas.set_image(None)


    def reset_overlays_and_segmentation(self, full_reset_models=False):
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.clear_all_overlays()

        if full_reset_models or self.cv_image_rgb_orig is None:
            self.bgd_model = np.zeros((1, 65), dtype=np.float64)
            self.fgd_model = np.zeros((1, 65), dtype=np.float64)
            self.cv_mask = None
            self.is_segmented = False
        elif self.cv_image_rgb_orig is not None:
             self.cv_mask = np.zeros(self.cv_image_rgb_orig.shape[:2], dtype=np.uint8)

        if self.cv_image_rgb_orig is not None:
            self.cv_image_rgb_display = self.cv_image_rgb_orig.copy()
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.set_image(self.cv_image_rgb_display)
        elif hasattr(self, 'canvas') and self.canvas:
            self.canvas.set_image(None)
            
        self.save_button.setEnabled(self.is_segmented and not full_reset_models) 
        
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.update_display_pixmap()

    def run_segmentation_process(self):
        if self.cv_image_rgb_orig is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала загрузите изображение.")
            return

        rect_cv, fg_strokes, bg_strokes = self.canvas.get_grabcut_params()
        
        current_grabcut_mode = cv2.GC_EVAL

        if not self.is_segmented or self.cv_mask is None or np.all(self.bgd_model == 0):
            self.bgd_model = np.zeros((1,65),np.float64)
            self.fgd_model = np.zeros((1,65),np.float64)
            self.cv_mask = np.zeros(self.cv_image_rgb_orig.shape[:2],np.uint8)

            if rect_cv:
                if rect_cv[2] <= 0 or rect_cv[3] <= 0:
                    QMessageBox.warning(self, "Ошибка ROI", "Прямоугольник ROI имеет некорректные размеры.")
                    return
                current_grabcut_mode = cv2.GC_INIT_WITH_RECT
            elif fg_strokes or bg_strokes:
                rect_cv_full_image = (1,1, self.cv_image_rgb_orig.shape[1]-2, self.cv_image_rgb_orig.shape[0]-2)
                if rect_cv_full_image[2] <=0 or rect_cv_full_image[3] <=0:
                    QMessageBox.warning(self, "Ошибка изображения", "Изображение слишком мало для инициализации мазками.")
                    return

                for stroke in bg_strokes: # ИСПРАВЛЕНО ЗДЕСЬ
                    if len(stroke)>1:
                        points_array = np.array([[p.x(), p.y()] for p in stroke], dtype=np.int32)
                        cv2.polylines(self.cv_mask,[points_array],False,cv2.GC_BGD,5)
                for stroke in fg_strokes: # И ИСПРАВЛЕНО ЗДЕСЬ
                    if len(stroke)>1:
                        points_array = np.array([[p.x(), p.y()] for p in stroke], dtype=np.int32)
                        cv2.polylines(self.cv_mask,[points_array],False,cv2.GC_FGD,5)
                
                current_grabcut_mode = cv2.GC_INIT_WITH_MASK
                rect_cv = rect_cv_full_image
            else:
                QMessageBox.warning(self, "Нет разметки", "Нарисуйте ROI или мазки для инициализации GrabCut.")
                return
        
        elif self.is_segmented and (fg_strokes or bg_strokes):
            current_grabcut_mode = cv2.GC_EVAL
            for stroke in bg_strokes: # ИСПРАВЛЕНО ЗДЕСЬ
                if len(stroke)>1:
                    points_array = np.array([[p.x(), p.y()] for p in stroke], dtype=np.int32)
                    cv2.polylines(self.cv_mask,[points_array],False,cv2.GC_BGD,5)
            for stroke in fg_strokes: # И ИСПРАВЛЕНО ЗДЕСЬ
                if len(stroke)>1:
                    points_array = np.array([[p.x(), p.y()] for p in stroke], dtype=np.int32)
                    cv2.polylines(self.cv_mask,[points_array],False,cv2.GC_FGD,5)
            
            self.canvas.fg_strokes_list.clear()
            self.canvas.bg_strokes_list.clear()


        iterations = self.iter_spinbox.value()
        try:
            print(f"Запуск GrabCut: mode={current_grabcut_mode}, rect={rect_cv}, iters={iterations}")
            
            if current_grabcut_mode == cv2.GC_INIT_WITH_RECT and rect_cv is None:
                 QMessageBox.critical(self, "Ошибка GrabCut", "Для режима GC_INIT_WITH_RECT необходим ROI.")
                 return
            if rect_cv is not None and (rect_cv[0] < 0 or rect_cv[1] < 0 or \
                rect_cv[0] + rect_cv[2] > self.cv_image_rgb_orig.shape[1] or \
                rect_cv[1] + rect_cv[3] > self.cv_image_rgb_orig.shape[0]):
                QMessageBox.warning(self, "Ошибка ROI", "Прямоугольник ROI выходит за пределы изображения.")
                # Не будем сбрасывать, дадим пользователю исправить ROI
                return


            cv2.grabCut(self.cv_image_rgb_orig, self.cv_mask, rect_cv, 
                        self.bgd_model, self.fgd_model,
                        iterations, current_grabcut_mode)
            
            self.is_segmented = True
            self.save_button.setEnabled(True)
        
        except cv2.error as e_cv:
            QMessageBox.critical(self, "Ошибка GrabCut (cv2.error)", f"Ошибка OpenCV: {e_cv}\nУбедитесь, что ROI валиден.")
            self.reset_overlays_and_segmentation(full_reset_models=True)
            return
        except Exception as e_generic:
            QMessageBox.critical(self, "Ошибка GrabCut", f"Непредвиденная ошибка: {e_generic}")
            self.reset_overlays_and_segmentation(full_reset_models=True)
            return

        self.cv_image_rgb_display = self.cv_image_rgb_orig.copy()
        mask_is_fg_or_prfg = np.where((self.cv_mask == cv2.GC_FGD) | (self.cv_mask == cv2.GC_PR_FGD), True, False)
        self.cv_image_rgb_display[~mask_is_fg_or_prfg] = [0, 0, 0] 

        self.canvas.set_image(self.cv_image_rgb_display)
        
        if current_grabcut_mode == cv2.GC_INIT_WITH_MASK: # Также очищаем мазки, если была инициализация по ним
            self.canvas.fg_strokes_list.clear()
            self.canvas.bg_strokes_list.clear()
            self.canvas.update_display_pixmap() 
        
        QMessageBox.information(self, "GrabCut", "Сегментация завершена.")


    def save_segmentation_result(self):
        if not self.is_segmented or self.cv_image_rgb_orig is None or self.cv_mask is None:
            QMessageBox.warning(self, "Нет данных", "Сначала выполните сегментацию.")
            return

        grabcut_results_dir = os.path.join(self.output_base_dir, "grabcut_interactive_results")
        if not os.path.exists(grabcut_results_dir):
            try:
                os.makedirs(grabcut_results_dir)
            except Exception as e:
                 QMessageBox.critical(self, "Ошибка создания папки", f"Не удалось создать папку: {grabcut_results_dir}\n{e}")
                 return

        base_filename, orig_ext = os.path.splitext(self.original_filename_for_saving)
        if not orig_ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            orig_ext = ".png"
        
        save_path_segmented_img = os.path.join(grabcut_results_dir, f"{base_filename}_segmented{orig_ext}")
        try:
            cv2.imwrite(save_path_segmented_img, cv2.cvtColor(self.cv_image_rgb_display, cv2.COLOR_RGB2BGR))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить сегментированное изображение: {e}")
            return

        binary_mask_to_save = np.where((self.cv_mask == cv2.GC_FGD) | (self.cv_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        save_path_binary_mask = os.path.join(grabcut_results_dir, f"{base_filename}_mask.png")
        try:
            cv2.imwrite(save_path_binary_mask, binary_mask_to_save)
        except Exception as e:
             QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить бинарную маску: {e}")
             return

        QMessageBox.information(self, "Сохранено", f"Результаты сохранены в:\nСегментированное: {save_path_segmented_img}\nМаска: {save_path_binary_mask}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    test_output_directory = "grabcut_tool_standalone_output" 
    if not os.path.exists(test_output_directory):
        try:
            os.makedirs(test_output_directory)
            print(f"Создана тестовая папка: {test_output_directory}")
        except OSError as e:
            print(f"Ошибка создания тестовой папки '{test_output_directory}': {e}")
            
    dialog = GrabCutDialog(output_base_dir=test_output_directory)
    dialog.show()
    sys.exit(app.exec())