import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QStackedWidget, QLabel, QLineEdit, QFileDialog, QGroupBox, QGridLayout,
    QComboBox, QMessageBox, QTextEdit, QProgressBar, QRadioButton, QSpinBox,
    QSizePolicy, QFrame, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor, QIcon, QTextCursor
from PyQt5.QtCore import Qt, QSize, QThreadPool, QObject, QRunnable, pyqtSignal, pyqtSlot, QTimer
from rdkit import Chem
from rdkit.Chem import Draw
from utils_GUI import PredictionController, process_scaffold_smi, run_generation, run_generation_and_prediction


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.kwargs['status_callback'] = lambda s: self.signals.status.emit(str(s))
        self.kwargs['progress_callback'] = lambda p: self.signals.progress.emit(int(p))

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            import traceback
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


def draw_molecule_to_pixmap(smiles, size=(300, 300)):
    if not smiles or smiles == "Input SMILES String...":
        pixmap = QPixmap(size[0], size[1])
        pixmap.fill(Qt.white)
        return pixmap

    smi_for_drawing = smiles.replace('[R]', '[*]')
    mol = Chem.MolFromSmiles(smi_for_drawing)
    if not mol:
        pixmap = QPixmap(size[0], size[1])
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setPen(Qt.red)
        painter.setFont(QFont("Arial", 10))
        rect = pixmap.rect().adjusted(5, 5, -5, -5)
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, "Invalid SMILES")
        painter.end()
        return pixmap

    img = Draw.MolToImage(mol, size=size)
    qimage = QImage(img.tobytes("raw", "RGBA"), img.width, img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimage)


GROUPBOX_STYLE = """
        QGroupBox#BlueTitleGroup {
            font-weight: bold;
            border: 2px solid #E5E7EB;
            border-radius: 15px;
            margin-top: 25px;
            background-color: #F9FAFB; 
        }
        QGroupBox#BlueTitleGroup::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 20px;
            padding: 8px 25px;
            background-color: #1B5EA1;
            color: white;
            border-radius: 12px;
        }
    """


class ResultCard(QWidget):
    def __init__(self, header="", value="?", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header_label = QLabel(header)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet(
            "background-color: #1B5EA1; color: white; font-size: 22px; "
            "padding: 8px 16px; border-radius: 6px 6px 0 0; font-weight: bold;"
        )
        self.value_label = QLabel(value)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setMinimumWidth(200)
        self.value_label.setMinimumHeight(70)
        self.set_value(value, "default")
        layout.addWidget(self.header_label)
        layout.addWidget(self.value_label)

    def set_value(self, value, mode="default"):
        self.value_label.setText(value)
        base_style = "font-size: 28px; font-weight: bold; padding: 20px; border: 2px solid #1B5EA1; border-top: none; border-radius: 0 0 6px 6px; "

        if mode == "default":
            self.value_label.setStyleSheet(base_style + "background-color: #F3F4F6; color: #9CA3AF;")
        elif mode == "yes":
            self.value_label.setStyleSheet(base_style + "background-color: rgba(27, 94, 161, 0.15); color: #1B5EA1;")
        elif mode == "no":
            self.value_label.setStyleSheet(
                base_style + "background-color: rgba(220, 38, 38, 0.1); color: rgba(220, 38, 38, 0.7); border: 0.5px solid rgba(220, 38, 38, 0.3);")
        else:
            self.value_label.setStyleSheet(base_style + "background-color: #FFFFFF; color: #1B5EA1;")


class MainPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 80, 50, 80)
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignCenter)

        # Header: Logo (Top) + Title (Bottom)
        header_container = QWidget()
        header_vbox = QVBoxLayout(header_container)
        header_vbox.setAlignment(Qt.AlignCenter)
        header_vbox.setSpacing(10)

        logo_label = QLabel()
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            logo_label.setPixmap(QPixmap(logo_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        title_label = QLabel("Intelligent Design of Type I Photosensitizers")
        title_label.setStyleSheet(
            "font-size: 22pt; font-weight: bold; color: #1B5EA1; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)

        header_vbox.addWidget(logo_label, alignment=Qt.AlignCenter)
        header_vbox.addWidget(title_label, alignment=Qt.AlignCenter)

        subtitle = QLabel("Select a task to begin")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18pt; color: #6B7280; background: transparent;")

        # Task Buttons
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setSpacing(40)
        btn_layout.setContentsMargins(0, 30, 0, 0)
        btn_container.setFixedWidth(500)

        self.btn_pred = self._make_task_btn("Prediction", "#1B5EA1", "#2563EB", "#1E40AF")
        self.btn_gen = self._make_task_btn("Generation", "#2E7D32", "#16A34A", "#14532D")
        self.btn_comb = self._make_task_btn("Generation + Prediction", "#6A1B9A", "#9333EA", "#581C87")
        self.setStyleSheet("background-color: white;")

        btn_layout.addWidget(self.btn_pred)
        btn_layout.addWidget(self.btn_gen)
        btn_layout.addWidget(self.btn_comb)

        layout.addWidget(header_container)
        layout.addWidget(subtitle)
        layout.addWidget(btn_container, alignment=Qt.AlignCenter)

        self.btn_pred.clicked.connect(lambda: main_window.show_page(1))
        self.btn_gen.clicked.connect(lambda: main_window.show_page(2))
        self.btn_comb.clicked.connect(lambda: main_window.show_page(3))

    def _make_task_btn(self, text, base, hover, pressed):
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
                QPushButton {{ 
                    background-color: {base}; 
                    color: white; 
                    border-radius: 15px; 
                    border: none;
                    font-size: 18pt;   
                    font-weight: bold;
                }}
                QPushButton:hover {{ background-color: {hover}; }}
                QPushButton:pressed {{ background-color: {pressed}; }}
            """)

        btn.setMinimumHeight(75)
        return btn


class BasePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = QThreadPool()

    def create_back_button(self):
        btn = QPushButton("← Back to Main Page")
        btn.setObjectName("BackButton")
        btn.setFixedWidth(200)
        btn.setStyleSheet("font-weight: bold;")
        btn.clicked.connect(self.main_window.reset_all_and_home)
        return btn


class PredictionPage(BasePage):
    DEFAULT_SMILES = "CN(C)c1ccc2c(c1)CCc1cc3ccc(N(C)C)cc3[s+]c1-2"

    def __init__(self, main_window):
        super().__init__(main_window)
        self._current_csv_df = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        layout.addWidget(self.create_back_button())

        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(30)

        left_col = QVBoxLayout()
        left_col.setSpacing(15)

        input_group = QGroupBox("Input")
        input_group.setObjectName("BlueTitleGroup")
        input_grid = QGridLayout(input_group)
        input_grid.setSpacing(10)

        self.radio_single = QRadioButton("Single SMILES")
        self.radio_csv    = QRadioButton("CSV File")
        self.radio_single.setChecked(True)
        self.smiles_input = QLineEdit(self.DEFAULT_SMILES)
        self.csv_path = QLineEdit("Select Input Path...")
        self.csv_path.setReadOnly(True)
        self.btn_browse_csv = QPushButton("Browse...")
        self.btn_browse_csv.setFixedWidth(100)
        self.lbl_sample = QLabel("Preview Sample:")
        self.spin_sample = QSpinBox()
        self.spin_sample.setMinimum(1)
        self.spin_sample.setValue(1)
        self.spin_sample.setEnabled(False)
        input_grid.addWidget(self.radio_single, 0, 0)
        input_grid.addWidget(self.smiles_input, 0, 1, 1, 2)
        input_grid.addWidget(self.radio_csv, 1, 0)
        input_grid.addWidget(self.csv_path, 1, 1)
        input_grid.addWidget(self.btn_browse_csv, 1, 2)
        input_grid.addWidget(self.lbl_sample, 2, 0)
        input_grid.addWidget(self.spin_sample, 2, 1)
        left_col.addWidget(input_group)

        output_group = QGroupBox("Output")
        output_group.setObjectName("BlueTitleGroup")
        output_layout = QHBoxLayout(output_group)
        self.output_path = QLineEdit("Select Output Path...")
        self.output_path.setReadOnly(True)
        self.btn_browse_out = QPushButton("Browse...")
        self.btn_browse_out.setFixedWidth(120)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.btn_browse_out)
        left_col.addWidget(output_group)

        self.btn_run = QPushButton("Run Prediction")
        self.btn_run.setMinimumHeight(45)
        self.btn_run.setStyleSheet("background-color: #1B5EA1; font-size: 12pt;")
        left_col.addWidget(self.btn_run)

        right_col = QVBoxLayout()
        right_col.setSpacing(15)
        preview_group = QGroupBox("Structure Preview")
        preview_group.setObjectName("BlueTitleGroup")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(15, 20, 15, 20) # left, top, right, bottom
        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(300, 220) # W, H
        self.lbl_preview.setStyleSheet("""background-color: white; border-radius: 6px; border: 1px solid #E5E7EB;""")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.lbl_preview, alignment=Qt.AlignCenter)
        right_col.addWidget(preview_group)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setObjectName("ClearAllButton")
        self.btn_clear.setMinimumHeight(45)
        self.btn_clear.setStyleSheet("background-color: #9CA3AF; font-size: 12pt;")
        right_col.addWidget(self.btn_clear)
        mid_layout.addLayout(left_col, 3)
        mid_layout.addLayout(right_col, 2)
        layout.addLayout(mid_layout)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #4B5563;")
        self.progress = QProgressBar()
        self.progress.setFixedHeight(12)
        self.progress.setTextVisible(False)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress)

        results_group = QGroupBox("Results")
        results_group.setObjectName("BlueTitleGroup")

        results_outer_layout = QVBoxLayout(results_group)
        results_outer_layout.setContentsMargins(10, 30, 10, 15)
        results_outer_layout.setSpacing(5)

        cards_layout = QHBoxLayout()
        cards_layout.setAlignment(Qt.AlignCenter)
        cards_layout.setSpacing(50)

        self.card_single = ResultCard("Type I")
        self.card_unique = ResultCard("Unique Molecules")
        self.card_ratio = ResultCard("Type I (Ratio)")
        self.card_unique.hide()
        self.card_ratio.hide()

        cards_layout.addStretch()
        cards_layout.addWidget(self.card_single, alignment=Qt.AlignCenter)
        cards_layout.addWidget(self.card_unique, alignment=Qt.AlignCenter)
        cards_layout.addWidget(self.card_ratio, alignment=Qt.AlignCenter)
        cards_layout.addStretch()

        self.lbl_save_info = QLabel("")
        self.lbl_save_info.setAlignment(Qt.AlignCenter)
        self.lbl_save_info.setStyleSheet("color: #9CA3AF; font-size: 14px; background-color: transparent; padding-top: 5px;")

        results_outer_layout.addLayout(cards_layout)
        results_outer_layout.addWidget(self.lbl_save_info)

        layout.addWidget(results_group)

        self.radio_single.toggled.connect(self.on_input_toggle)
        self.smiles_input.textChanged.connect(self.update_preview)
        self.btn_browse_csv.clicked.connect(self.browse_csv)
        self.btn_browse_out.clicked.connect(self.browse_out)
        self.btn_clear.clicked.connect(self.reset_ui)
        self.btn_run.clicked.connect(self.run_task)
        self.spin_sample.valueChanged.connect(self.on_spin_changed)
        self.update_preview()
        self.on_input_toggle()

        self.setStyleSheet(GROUPBOX_STYLE)

        self.smiles_input.textEdited.connect(self.clear_results_and_status)

    def reset_ui(self):
        self.radio_single.blockSignals(True)
        self.radio_single.setChecked(True)
        self.radio_single.blockSignals(False)
        self.smiles_input.setText(self.DEFAULT_SMILES)
        self.csv_path.setText("Select Input Path...")
        self.output_path.setText("Select Output Path...")
        self._current_csv_df = None
        self.progress.setValue(0)
        self.spin_sample.setValue(1)
        self.lbl_status.setText("Status: Ready")
        self.card_single.set_value("?", "default")
        self.card_unique.set_value("?", "default")
        self.card_ratio.set_value("?", "default")
        self.on_input_toggle()
        self.update_preview()
        self.spin_sample.setEnabled(False)
        self.lbl_save_info.setText("")

    def on_input_toggle(self):
        self.clear_results_and_status()

        self.spin_sample.blockSignals(True)
        self.spin_sample.setValue(1)
        self.spin_sample.blockSignals(False)

        is_single = self.radio_single.isChecked()
        self.smiles_input.setEnabled(is_single)
        self.csv_path.setEnabled(not is_single)
        self.btn_browse_csv.setEnabled(not is_single)
        active_style = "background-color: #1B5EA1; color: white; border-radius: 5px;"
        inactive_style = "background-color: #D1D5DB; color: #4B5563; border-radius: 5px;"
        self.card_ratio.hide()
        self.spin_sample.setEnabled(False)

        self.output_path.setText("Select Output Path...")
        self.card_single.set_value("?", "default")
        self.card_unique.set_value("?", "default")
        self.card_ratio.set_value("?", "default")
        self.progress.setValue(0)
        self.lbl_status.setText("Status: Ready")
        self.lbl_save_info.setText("")
        self.btn_browse_out.setEnabled(True)
        self.btn_browse_out.setStyleSheet(active_style)

        if is_single:
            self.btn_browse_csv.setStyleSheet(inactive_style)
            self.btn_browse_csv.setDisabled(True)
            self.spin_sample.setEnabled(False)

            self.smiles_input.setText(self.DEFAULT_SMILES)
            self.csv_path.setText("Select Input Path...")
            self._current_csv_df = None
            self.card_single.show()
            self.card_unique.hide()
            self.card_ratio.hide()
        else:
            self.btn_browse_csv.setStyleSheet(active_style)
            self.btn_browse_csv.setEnabled(True)
            self.smiles_input.setText("Input SMILES String...")
            self.card_single.hide()
            self.card_unique.show()
            self.card_ratio.show()
            self.spin_sample.setEnabled(self._current_csv_df is not None)
        self.update_preview()

    def update_preview(self):
        smi = ""
        if self.radio_single.isChecked():
            smi = self.smiles_input.text()
            if smi == "Input SMILES String...":
                smi = ""
        else:
            if self._current_csv_df is not None:
                idx = self.spin_sample.value() - 1
                if 0 <= idx < len(self._current_csv_df):
                    cols = self._current_csv_df.columns
                    s_cols = [c for c in cols if 'smiles' in c.lower()]
                    if s_cols:
                        smi = str(self._current_csv_df[s_cols[0]].iloc[idx])
        self.lbl_preview.setPixmap(draw_molecule_to_pixmap(smi, size=(300, 220)))

    def browse_csv(self):
        self.clear_results_and_status()
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.csv_path.setText(path)
            try:
                self._current_csv_df = pd.read_csv(path)
                self.spin_sample.setMaximum(len(self._current_csv_df))
                self.spin_sample.setValue(1)
                self.spin_sample.setEnabled(True)
                self.update_preview()
            except:
                QMessageBox.warning(self, "Error", "Failed to Load CSV File.")
                self.spin_sample.setEnabled(False)

    def browse_out(self):
        self.lbl_save_info.setText("")
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            self.output_path.setText(path)

    def run_task(self):
        predictor = self.main_window.predictor
        if not predictor.is_ready:
            QMessageBox.warning(
                self,
                "Warning",
                f"The prediction function is unavailable due to a missing prediction file!\n\nReason: \n{predictor.load_error_msg}"
            )
            return

        self.clear_results_and_status()
        self.skipped_count = 0

        if self.radio_single.isChecked():
            smi = self.smiles_input.text().strip()
            if not smi or smi == "Input SMILES String...":
                QMessageBox.warning(self, "Input Error", "Please Input a SMILES String!")
                return
            if Chem.MolFromSmiles(smi) is None:
                QMessageBox.warning(self, "Invalid SMILES", "The Provided SMILES String is Invalid.")
                return

            smiles = [smi]
        else:
            if self._current_csv_df is None or self._current_csv_df.empty:
                QMessageBox.warning(self, "Input Error", "Please Upload a CSV File!")
                return

            cols = self._current_csv_df.columns
            s_cols = [c for c in cols if 'smiles' in c.lower()]
            if not s_cols:
                QMessageBox.warning(self, "Data Error", "No 'smiles' Column Found in the CSV File.")
                return

            raw_smiles = self._current_csv_df[s_cols[0]].dropna().astype(str).tolist()
            valid_smiles = [s for s in raw_smiles if Chem.MolFromSmiles(s) is not None]

            self.skipped_count = len(raw_smiles) - len(valid_smiles)

            if not valid_smiles:
                QMessageBox.warning(self, "Data Error", "No Valid SMILES Strings Found in the CSV File.")
                return

            smiles = valid_smiles

        out = self.output_path.text()
        if out == "Select Output Path..." or not out.strip():
            out = None
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Status: Starting Prediction...")
        worker = Worker(self.main_window.predictor.run_prediction, smiles_list=smiles, output_path=out)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.result.connect(self.on_result)
        worker.signals.status.connect(lambda s: self.lbl_status.setText(f"Status: {s}"))
        worker.signals.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.threadpool.start(worker)

    def on_result(self, res):
        self.progress.setValue(100)
        status_text = "Status: Prediction Completed!"
        if hasattr(self, 'skipped_count') and self.skipped_count > 0:
            highlight_html = f" <span style='color: #1B5EA1; font-weight: bold;'>(Skipped {self.skipped_count} invalid SMILES).</span>"
            self.lbl_status.setText(status_text + highlight_html)
        else:
            self.lbl_status.setText(status_text)

        path_text = self.output_path.text()
        if path_text == "Select Output Path...":
            save_info = "Result not Saved."
        else:
            save_info = f"Result Saved to: {path_text}"

        self.lbl_save_info.setText(save_info)

        if self.radio_single.isChecked():
            val = res.get('single_result')
            if val == 1: self.card_single.set_value("Yes", "yes")
            else: self.card_single.set_value("No", "no")
        else:
            self.card_unique.set_value(str(res['total_unique']), "plain")
            self.card_ratio.set_value(f"{res['count_type1']} ({res['percentage_type1']:.1f}%)", "plain")

    def on_spin_changed(self):
        if self.radio_csv.isChecked() and self._current_csv_df is None:
            QMessageBox.warning(self, "Warning", "Please Upload a CSV File!")
            self.spin_sample.blockSignals(True)
            self.spin_sample.setValue(1)
            self.spin_sample.blockSignals(False)
            return
        self.update_preview()

    def clear_results_and_status(self):
        self.card_single.set_value("?", "default")
        self.card_unique.set_value("?", "default")
        self.card_ratio.set_value("?", "default")
        self.lbl_status.setText("Status: Ready")
        self.lbl_save_info.setText("")
        self.progress.setValue(0)


class GenerationPage(BasePage):
    DEFAULT_SCAFFOLD = "[R]C1C=C2CCC3C(C2=CC=1)=[S+]C1C(=CC=C(C=1)[R])C=3"

    def __init__(self, main_window, is_comb=False):
        super().__init__(main_window)
        self.is_comb = is_comb
        self._last_line_is_stage = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(10)

        layout.addWidget(self.create_back_button())

        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(30)

        left_col = QVBoxLayout()
        left_col.setSpacing(15)

        input_group = QGroupBox("Scaffold Input")
        input_group.setObjectName("BlueTitleGroup")
        input_grid = QGridLayout(input_group)
        input_grid.setContentsMargins(15, 20, 15, 15)  #left, top, right, down

        self.txt_scaf = QLineEdit(self.DEFAULT_SCAFFOLD)
        input_grid.addWidget(QLabel("Scaffold SMILES:"), 0, 0)
        input_grid.addWidget(self.txt_scaf, 0, 1)
        left_col.addWidget(input_group)

        params_group = QGroupBox("Generation Parameters")
        params_group.setObjectName("BlueTitleGroup")
        params_grid = QGridLayout(params_group)

        self.cmb_rand = QComboBox()
        self.cmb_rand.addItems(['2', '4', '6', '8', '16'])
        self.cmb_rand.setCurrentText('8')

        self.cmb_dec = QComboBox()
        self.cmb_dec.addItems(['4', '6', '8', '16', '32'])
        self.cmb_dec.setCurrentText('16')

        params_grid.addWidget(QLabel("Randomized Scaffold SMILES:"), 0, 0)
        params_grid.addWidget(self.cmb_rand, 0, 1)
        params_grid.addWidget(QLabel("Decorations per Scaffold SMILES:"), 1, 0)
        params_grid.addWidget(self.cmb_dec, 1, 1)
        left_col.addWidget(params_group)

        output_group = QGroupBox("Output Root Path")
        output_group.setObjectName("BlueTitleGroup")
        output_layout = QHBoxLayout(output_group)
        output_layout.setContentsMargins(15, 20, 15, 15)

        self.txt_out = QLineEdit("Select Output Root Path...")
        self.txt_out.setReadOnly(True)
        self.btn_dir = QPushButton("Browse...")
        self.btn_dir.setFixedWidth(100)
        output_layout.addWidget(self.txt_out)
        output_layout.addWidget(self.btn_dir)
        left_col.addWidget(output_group)

        run_color = "#2E7D32" if not is_comb else "#6A1B9A"
        self.btn_run = QPushButton("Run Generation and Prediction" if is_comb else "Run Generation")
        self.btn_run.setMinimumHeight(45)
        self.btn_run.setStyleSheet(
            f"background-color: {run_color}; font-size: 12pt; color: white; font-weight: bold; border-radius: 5px;")
        left_col.addWidget(self.btn_run)

        right_col = QVBoxLayout()
        right_col.setSpacing(15)

        preview_group = QGroupBox("Scaffold Preview")
        preview_group.setObjectName("BlueTitleGroup")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(15, 20, 15, 20)
        self.lbl_prev = QLabel()
        self.lbl_prev.setFixedSize(300, 220)
        self.lbl_prev.setStyleSheet("background-color: white; border-radius: 6px; border: 1px solid #E5E7EB;")
        self.lbl_prev.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.lbl_prev, alignment=Qt.AlignCenter)
        right_col.addWidget(preview_group)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setMinimumHeight(45)
        self.btn_clear.setStyleSheet(
            "background-color: #9CA3AF; font-size: 12pt; color: white; font-weight: bold; border-radius: 5px;")
        right_col.addWidget(self.btn_clear)

        mid_layout.addLayout(left_col, 3)
        mid_layout.addLayout(right_col, 2)
        layout.addLayout(mid_layout)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #4B5563;")
        self.progress = QProgressBar()
        self.progress.setFixedHeight(12)
        self.progress.setTextVisible(False)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress)

        results_group = QGroupBox("Results")
        results_group.setObjectName("BlueTitleGroup")
        results_outer_layout = QVBoxLayout(results_group)
        results_outer_layout.setContentsMargins(15, 25, 15, 1)  #left, top, right, down

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        self.result_text.setStyleSheet("""
                   QTextEdit {
                       background-color: #FFFFFF;
                       border: 1px solid #E5E7EB;
                       border-radius: 5px;
                       font-family: 'Consolas', 'Monaco', monospace;
                       font-size: 12px;
                       color: #374151;
                   }
               """)
        results_outer_layout.addWidget(self.result_text)

        self.lbl_save_info = QLabel("")
        self.lbl_save_info.setAlignment(Qt.AlignCenter)
        self.lbl_save_info.setStyleSheet("color: #9CA3AF; font-size: 13px; padding-top: 5px;")
        results_outer_layout.addWidget(self.lbl_save_info)

        layout.addWidget(results_group, stretch=1)

        self.txt_scaf.textChanged.connect(self.update_preview)
        self.txt_scaf.textEdited.connect(self.clear_results_and_status)
        self.btn_dir.clicked.connect(self.browse_dir)
        self.btn_clear.clicked.connect(self.reset_ui)
        self.btn_run.clicked.connect(self.run_task)

        self.update_preview()
        self.setStyleSheet(GROUPBOX_STYLE)

    def clear_results_and_status(self):
        self.lbl_status.setText("Status: Ready")
        self.result_text.clear()
        self.lbl_save_info.setText("")
        self.progress.setValue(0)
        self._last_line_is_stage = False

    def reset_ui(self):
        self.txt_scaf.setText(self.DEFAULT_SCAFFOLD)
        self.cmb_rand.setCurrentText('8')
        self.cmb_dec.setCurrentText('16')
        self.txt_out.setText("Select Output Root Path...")
        self.lbl_status.setText("Status: Ready")
        self.result_text.clear()
        self.progress.setValue(0)
        self.update_preview()

    def update_preview(self):
        smi = self.txt_scaf.text().strip()
        self.lbl_prev.setPixmap(draw_molecule_to_pixmap(smi, size=(300, 220)))

    def browse_dir(self):
        self.clear_results_and_status()
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.txt_out.setText(path)

    def run_task(self):
        smi = self.txt_scaf.text().strip()

        if not smi:
            QMessageBox.warning(self, "Input Error", "Please Input a Scaffold SMILES String!")
            return

        if self.is_comb:
            predictor = self.main_window.predictor
            if not predictor.is_ready:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "The combined task is unavailable because the prediction model is not ready!\n\nYou can still return to the main page to use the Generation function."
                )
                return

        has_r = '[R]' in smi
        has_star = '[*]' in smi
        has_indexed_star = '[*:' in smi

        if not (has_r or has_star or has_indexed_star):
            QMessageBox.warning(self, "Attachment Error",
                                "Please Specify the Attachment Point (e.g., [R], [*], [*:0])!")
            return

        if Chem.MolFromSmiles(smi.replace('[R]', '[*]')) is None:
            if Chem.MolFromSmiles(smi) is None:
                QMessageBox.warning(self, "Invalid SMILES", "The Scaffold SMILES is Invalid!")
                return

        try:
            processed_smi = process_scaffold_smi(smi)
            print(f"Processed SMILES for Model: {processed_smi}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"SMILES Processing Error: {str(e)}")
            return

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Status: Starting Generation...")
        self.result_text.clear()
        self.progress.setRange(0, 0)

        if self.is_comb: # generation + prediction
            worker = Worker(
                run_generation_and_prediction,
                predictor_obj=self.main_window.predictor,
                scaffold_smi=processed_smi,
                output_dir=self.txt_out.text() if self.txt_out.text() != "Select Output Root Path..." else ".",
                num_random=int(self.cmb_rand.currentText()),
                num_decor=int(self.cmb_dec.currentText())
            )
        else:
            worker = Worker(
                run_generation,
                scaffold_smi=processed_smi,
                output_dir=self.txt_out.text() if self.txt_out.text() != "Select Output Root Path..." else ".",
                num_random=int(self.cmb_rand.currentText()),
                num_decor=int(self.cmb_dec.currentText())
            )

        worker.signals.status.connect(self.append_log)
        worker.signals.result.connect(self.on_done)
        worker.signals.error.connect(lambda e: QMessageBox.critical(self, "Run Error", str(e[1])))
        worker.signals.finished.connect(lambda: self.btn_run.setEnabled(True))

        self.threadpool.start(worker)

    def on_done(self, result):
        self.btn_run.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        status_msg = "Combined Task Completed!" if self.is_comb else "Generation Completed!"
        self.lbl_status.setText(f"Status: {status_msg}")

        final_csv = os.path.abspath(result.get('prediction_csv_path', result['output_csv_path']))

        header = (
            "==================================================<br>"
            f"                  Task Completed<br>"
            "==================================================<br>"
            f"- Total Unique Molecules: {result['unique_count']}<br>"
        )
        footer = (
            f"- Top Molecules:           {result['top1_smiles']}<br>"
            f"- Result File:          {final_csv}<br>"
            "=================================================="
        )

        prediction_part = ""
        if self.is_comb:
            stats_style = "color: #1B5EA1; font-weight: bold; font-size: 16px;"
            prediction_part = (
                f"<br>- Predicted Type I Molecules (Ratio):      "
                f"<span style='{stats_style}'>{result['count_type1']} ({result['percentage_type1']:.1f}%)</span><br>"
            )

        final_html = f"<div style='font-family: Consolas, Monaco, monospace;'>{header}{footer}{prediction_part}</div>"
        self._last_line_is_stage = False

        self.result_text.append(final_html)
        self.result_text.ensureCursorVisible()

    def append_log(self, text):
        text = text.rstrip('\n').rstrip('\r')
        if not text.strip():
            return

        if "PID" in text:
            return

        is_progress = text.startswith("[Stage") or "Sampling:" in text

        cursor = self.result_text.textCursor()
        cursor.movePosition(QTextCursor.End)

        if is_progress and self._last_line_is_stage:
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        else:
            if not self.result_text.toPlainText() == "":
                cursor.insertBlock()
            cursor.insertText(text)

        self._last_line_is_stage = is_progress
        self.result_text.ensureCursorVisible()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSGenerator")
        self.setFixedSize(1000, 800)
        self.move(400, 100)

        if os.path.exists("style.qss"):
            with open("style.qss", "r") as f: self.setStyleSheet(f.read())

        if os.path.exists("logo.png"):
            self.setWindowIcon(QIcon("logo.png"))

        self.predictor = PredictionController()

        self.stacked = QStackedWidget()
        self.page_main = MainPage(self)
        self.page_pred = PredictionPage(self)
        self.page_gen = GenerationPage(self, False)  # Generation, no prediction model needed.
        self.page_comb = GenerationPage(self, True)  # Generate + Predict; clicking "Run" will intercept.
        self.stacked.addWidget(self.page_main)  # 0
        self.stacked.addWidget(self.page_pred)  # 1
        self.stacked.addWidget(self.page_gen)  # 2
        self.stacked.addWidget(self.page_comb)  # 3
        self.setCentralWidget(self.stacked)

    def show_page(self, idx): self.stacked.setCurrentIndex(idx)

    def show_main_page(self): self.show_page(0)

    def reset_all_and_home(self):
        self.page_pred.reset_ui()
        self.page_gen.reset_ui()
        self.page_comb.reset_ui()
        self.show_page(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
