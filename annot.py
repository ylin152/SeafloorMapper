import os.path
import sys, math
import socket
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QDialog, \
    QTableWidgetItem, QVBoxLayout, QMessageBox, QLabel, QLineEdit, QPushButton, QSizePolicy
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QCursor, QPixmap, QStandardItemModel, QStandardItem
from PyQt6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Patch
import matplotlib.ticker as ticker
from AnnotWindow import Ui_AntWindow
from AttributionWindow import Ui_AttributionWindow
from PreviewDialog import Ui_Dialog as Ui_PreviewDialog
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

font_path = font_manager.findfont(font_manager.FontProperties(family='Arial'))
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

base_dir = os.path.dirname(os.path.abspath(__file__))

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader(base_dir))

class InputDialog(QDialog):
    def __init__(self, default_text="", parent=None):
        super().__init__(parent)

        self.setWindowTitle("Set export folder")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.input_widget = QLineEdit(self)
        self.input_widget.setText(default_text)
        layout.addWidget(self.input_widget)

        confirm_button = QPushButton("Set", self)
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)

class FileDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open File")
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setNameFilters(["Text Files (*.txt *.csv)"])
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)

class FolderDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open folder")
        self.setFileMode(QFileDialog.FileMode.Directory)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)

class AttributionWindow(QMainWindow, Ui_AttributionWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class PreviewDialog(QDialog, Ui_PreviewDialog):
    selection_accepted = pyqtSignal(dict)

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        # self.setWindowTitle("Preview Data")

        # Set tabelWidget
        self.tableWidget.setRowCount(min(10, data.shape[0]))
        self.tableWidget.setColumnCount(data.shape[1])
        self.tableWidget.setHorizontalHeaderLabels([f"Column {i + 1}" for i in range(data.shape[1])])

        for row in range(self.tableWidget.rowCount()):
            for col in range(self.tableWidget.columnCount()):
                item = QTableWidgetItem(str(data[row, col]))
                self.tableWidget.setItem(row, col, item)

        # Retrieve column names
        col_names = [None]
        for col in range(self.tableWidget.columnCount()):
            col_name = self.tableWidget.horizontalHeaderItem(col).text()
            col_names.append(col_name)

        # Set combo box, let user select columns
        self.column_combos = {
            'lon': self.comboBox_lon,
            'lat': self.comboBox_lat,
            'y': self.comboBox_y,
            'elev': self.comboBox_elev,
            'conf': self.comboBox_conf,
            'label': self.comboBox_label,
        }

        for col_name, combo_box in self.column_combos.items():
            combo_box.addItems(col_names)

        if self.tableWidget.columnCount() == 8:
            self.comboBox_lon.setCurrentIndex(4)
            self.comboBox_lat.setCurrentIndex(5)
            self.comboBox_y.setCurrentIndex(2)
            self.comboBox_elev.setCurrentIndex(3)
            self.comboBox_conf.setCurrentIndex(7)
            self.comboBox_label.setCurrentIndex(8)

        # Initialize column selections
        self.column_indices = {
            'lon': None,
            'lat': None,
            'y': None,
            'elev': None,
            'conf': None,
            'label': None
        }

        # Plot the selected columns
        self.pushButton.clicked.connect(self.accept_selection)

    def accept_selection(self):
        for col_name, combo_box in self.column_combos.items():
            index = combo_box.currentIndex()

            if index > 0:
                self.column_indices[col_name] = index - 1

        if self.column_indices['y'] is None or self.column_indices['elev'] is None:
            message_box = QMessageBox(self)
            message_box.setWindowTitle("Error")
            message_box.setText("No column has been set for y distance and/or elevation, please check.")
            message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            message_box.setIcon(QMessageBox.Icon.Warning)
            message_box.exec()

            return

        if self.column_indices['lon'] is None or self.column_indices['lat'] is None:
            message_box = QMessageBox(self)
            message_box.setWindowTitle("Warning")
            message_box.setText("No column has been set for longitude and/or latitude, are you sure to proceed?")
            message_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            message_box.setIcon(QMessageBox.Icon.Question)
            button = message_box.exec()

            if button == QMessageBox.StandardButton.No:
                return

        # Send signal to plot_data
        self.selection_accepted.emit(self.column_indices)

        self.close()

def is_internet_available():
    try:
        socket.create_connection(('www.github.com', 80))
        return True
    except:
        return False


class AntWindow(QMainWindow, Ui_AntWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Initialize
        # File and Data
        self.init_dir = None
        self.root_dir = None
        self.output_dir = None
        self.model = None
        self.ext = None
        self.basename = None
        self.data = None
        self.data_copy = None
        self.columns = None
        self.column_indices = {}
        # self.type = 1
        # Canvas
        self.canvas = None
        self.figure = None
        self.scatter_ax = None
        self.scatter_ax2 = None
        self.scatter_plot = None
        self.current_cursor = Qt.CursorShape.ArrowCursor
        # Select tool
        self.select_flag = False
        self.selection_vertices = []
        self.selection_polygon = None
        self.selection_plot = None
        self.points_selected = []
        self.indices_selected = []
        self.last_indices_selected = []
        self.last_edit = None
        # Zoom tool
        self.origin = None
        self.y_interval = 50000
        self.y_range = ()
        self.elev_range = ()
        self.cur_elev_range = ()
        self.zoomin_x_flag = False
        self.zoomout_x_flag = False
        self.zoomin_y_flag = False
        self.zoomout_y_flag = False
        self.zoom_x_factor = 1.5
        self.zoom_y_factor = 1.1
        # Map
        self.browser = None
        self.lon_list = []
        self.lat_list = []
        self.coor = []
        self.lat_m = 0
        self.lon_m = 0
        self.coor_mean = []
        self.coor_selected = []
        # Save
        self.edit_flag = False

        # Set up canvas
        self.create_canvas()

        # Set up map
        if is_internet_available():
            self.create_map()
        else:
            label = QLabel('No Internet Access')
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout = QVBoxLayout()
            layout.addWidget(label)
            self.tab_map.setLayout(layout)

        # Add signal to "open file" in menu
        self.actionOpenFile.triggered.connect(self.open_file_dialog)
        self.actionOpenFolder.triggered.connect(self.open_folder_dialog)
        self.actionSave.triggered.connect(lambda: self.save_file_dialog(new=False))
        self.actionSave_As.triggered.connect(lambda: self.save_file_dialog(new=True))
        self.actionSetFolder.triggered.connect(self.set_export_folder)

        # Add signal to "about" in menu
        self.actionAttribution.triggered.connect(self.open_attr_window)

        # Plot control button group
        self.btn_zoominx.clicked.connect(self.zoom_in_x)
        self.btn_zoomoutx.clicked.connect(self.zoom_out_x)
        self.btn_zoominy.clicked.connect(self.zoom_in_y)
        self.btn_zoomouty.clicked.connect(self.zoom_out_y)
        self.btn_restore.clicked.connect(self.restore_plot)
        self.btn_select.clicked.connect(self.select_on_plot)

        # Label control button group
        self.button_add.clicked.connect(self.add_label)
        self.button_remove.clicked.connect(self.remove_label)
        self.button_undo.clicked.connect(self.undo_label)
        self.button_redo.clicked.connect(self.redo_label)
        self.button_clear.clicked.connect(self.clear_label)

        self.plot_widget.enterEvent = lambda event: self.enter_plot_widget(event)
        self.plot_widget.leaveEvent = lambda event: self.leave_plot_widget(event)

        self.scroll_bar.valueChanged.connect(self.on_scroll)

        self.fileView.doubleClicked.connect(self.open_selected_file)

    # File tree set-up
    def init_file_tree(self):
        self.model = QStandardItemModel()
        # root_path = r'C:\Users\Mac\Desktop\SeafloorMapper'

        if os.path.isfile(self.root_dir):
            file_item = QStandardItem(os.path.basename(self.root_dir))
            self.model.appendRow(file_item)
        else:
            folder_item = QStandardItem(os.path.basename(self.root_dir))
            self.model.appendRow(folder_item)
            for item_name in os.listdir(self.root_dir):
                # item_path = os.path.join(self.root_dir, item_name)
                item = QStandardItem(item_name)
                item.setToolTip(item_name)
                folder_item.appendRow(item)

        self.model.setHorizontalHeaderLabels(['Name'])
        self.fileView.setModel(self.model)
    
    # Canvas set-up
    def create_canvas(self):
        layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.canvas)
        # layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget.setLayout(layout)

        # Connect the mouse events for drawing the selection area
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    # Map set-up
    def create_map(self):
        self.browser = QWebEngineView()
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tab_map.setLayout(layout)

    # Interacting with plot
    def enter_plot_widget(self, event):
        self.plot_widget.setCursor(self.current_cursor)

    def leave_plot_widget(self, event):
        self.plot_widget.unsetCursor()

    def select_on_plot(self):
        if self.scatter_plot is not None:
            self.select_flag = True
            self.zoomin_x_flag = False
            self.zoomout_x_flag = False
            self.zoomin_y_flag = False
            self.zoomout_y_flag = False

            pixmap = QPixmap(os.path.join(base_dir,'icons/pencil.png'))
            self.current_cursor = QCursor(pixmap)

    def zoom_in_x(self):
        if self.scatter_plot is not None:
            self.zoomin_x_flag = True
            self.zoomout_x_flag = False
            self.zoomin_y_flag = False
            self.zoomout_y_flag = False
            self.select_flag = False

            pixmap = QPixmap(os.path.join(base_dir,'icons/magnifier-zoom-in.png'))
            self.current_cursor = QCursor(pixmap)

    def zoom_out_x(self):
        if self.scatter_plot is not None:
            self.zoomout_x_flag = True
            self.zoomin_x_flag = False
            self.zoomin_y_flag = False
            self.zoomout_y_flag = False
            self.select_flag = False

            pixmap = QPixmap(os.path.join(base_dir,'icons/magnifier-zoom-out.png'))
            self.current_cursor = QCursor(pixmap)

    def zoom_in_y(self):
        if self.scatter_plot is not None:
            self.zoomin_y_flag = True
            self.zoomout_x_flag = False
            self.zoomin_x_flag = False
            self.zoomout_y_flag = False
            self.select_flag = False

            pixmap = QPixmap(os.path.join(base_dir,'icons/magnifier-zoom-in.png'))
            self.current_cursor = QCursor(pixmap)

    def zoom_out_y(self):
        if self.scatter_plot is not None:
            self.zoomout_y_flag = True
            self.zoomin_x_flag = False
            self.zoomin_y_flag = False
            self.zoomout_x_flag = False
            self.select_flag = False

            pixmap = QPixmap(os.path.join(base_dir,'icons/magnifier-zoom-out.png'))
            self.current_cursor = QCursor(pixmap)

    def restore_plot(self):
        if self.scatter_plot is not None:
            self.current_cursor = Qt.CursorShape.ArrowCursor

            self.y_interval = 50000
            self.cur_elev_range = (self.elev_range[0] - 5, self.elev_range[1] + 5)
            self.scroll_bar.setMaximum(self.get_max_scroll())
            self.scroll_bar.setValue(0)
            self.plot_profile()

    # Draw on the canvas
    def on_mouse_press(self, event):
        if event.button == 1:  # left button OR if event.button is MouseButton.LEFT
            self.origin = (event.xdata, event.ydata)
            if self.select_flag:
                x_offset = 0.005 * (self.scatter_ax.get_xlim()[1] - self.scatter_ax.get_xlim()[0])
                y_offset = 0.01 * (self.scatter_ax.get_ylim()[1] - self.scatter_ax.get_ylim()[0])
                adjusted_vertices = (event.xdata - x_offset, event.ydata - y_offset)
                self.selection_vertices.append(adjusted_vertices)

                # self.selection_vertices.append((event.xdata, event.ydata))
            if self.zoomin_x_flag:
                x_limits = list(self.scatter_ax.get_xlim())
                x_dist = x_limits[1] - x_limits[0]
                # Define the new x limits
                if (self.origin[0] - (x_dist / self.zoom_x_factor) / 2) > 0:
                    x_limits[0] = self.origin[0] - (x_dist / self.zoom_x_factor) / 2
                x_limits[1] = self.origin[0] + (x_dist / self.zoom_x_factor) / 2
                # Define the new y dist interval
                self.y_interval = x_limits[1] - x_limits[0]
                self.scroll_bar.setMaximum(self.get_max_scroll())
                # Adjust the scroll bar value and update the profile
                self.scroll_bar.setValue(int((x_limits[0] - self.y_range[0]) / (self.y_interval / 2)))
                self.plot_profile()
            if self.zoomout_x_flag:
                x_limits = list(self.scatter_ax.get_xlim())
                x_dist = x_limits[1] - x_limits[0]
                x_limits[0] = self.origin[0] - (x_dist * self.zoom_x_factor) / 2
                x_limits[1] = self.origin[0] + (x_dist * self.zoom_x_factor) / 2
                self.y_interval = x_limits[1] - x_limits[0]
                self.scroll_bar.setMaximum(self.get_max_scroll())
                self.scroll_bar.setValue(int((x_limits[0] - self.y_range[0]) / (self.y_interval / 2)))
                self.plot_profile()
            if self.zoomin_y_flag:
                y_limits = list(self.scatter_ax.get_ylim())
                y_dist = y_limits[1] - y_limits[0]
                y_limits[0] = self.origin[1] - (y_dist / self.zoom_y_factor) / 2
                y_limits[1] = self.origin[1] + (y_dist / self.zoom_y_factor) / 2
                self.scatter_ax.set_ylim(y_limits[0], y_limits[1])
                self.cur_elev_range = (y_limits[0], y_limits[1])
                self.canvas.draw()
            if self.zoomout_y_flag:
                y_limits = list(self.scatter_ax.get_ylim())
                y_dist = y_limits[1] - y_limits[0]
                y_limits[0] = self.origin[1] - (y_dist * self.zoom_y_factor) / 2
                y_limits[1] = self.origin[1] + (y_dist * self.zoom_y_factor) / 2
                self.scatter_ax.set_ylim(y_limits[0], y_limits[1])
                self.cur_elev_range = (y_limits[0], y_limits[1])
                self.canvas.draw()

    def on_mouse_move(self, event):
        if event.button == 1:
            if self.select_flag:
                x_offset = 0.005 * (self.scatter_ax.get_xlim()[1] - self.scatter_ax.get_xlim()[0])
                y_offset = 0.01 * (self.scatter_ax.get_ylim()[1] - self.scatter_ax.get_ylim()[0])
                adjusted_vertices = (event.xdata - x_offset, event.ydata - y_offset)
                self.selection_vertices.append(adjusted_vertices)
                self.draw_selection_area(event)

    def on_mouse_release(self, event):
        if event.button == 1:
            if self.select_flag:
                x_offset = 0.005 * (self.scatter_ax.get_xlim()[1] - self.scatter_ax.get_xlim()[0])
                y_offset = 0.01 * (self.scatter_ax.get_ylim()[1] - self.scatter_ax.get_ylim()[0])
                adjusted_vertices = (event.xdata - x_offset, event.ydata - y_offset)
                self.selection_vertices.append(adjusted_vertices)
                self.draw_selection_area(event)
                # clear the previous selection vertices
                self.selection_vertices = []

    def draw_selection_area(self, event):
        # clear the previous selection polygon
        if self.selection_polygon is not None:
            self.selection_polygon.remove()
        if self.selection_plot is not None:
            self.selection_plot.remove()

        polygon = Polygon(self.selection_vertices, edgecolor="gray", facecolor="none", alpha=0.5)
        self.selection_polygon = self.scatter_ax.add_patch(polygon)
        self.select_points_in_area()
        self.canvas.draw()

    def select_points_in_area(self):
        # clear the previously selected points and indices
        self.points_selected = []
        self.indices_selected = []

        if self.scatter_plot is not None:
            polygon_path = self.selection_polygon.get_path()
            scatter_points = self.scatter_plot.get_offsets().data
            for i in range(len(scatter_points)):
                point = scatter_points[i]
                if polygon_path.contains_point(point):
                    self.points_selected.append(point)

                    y = point[0]
                    elev = point[1]
                    indices = np.where((self.data[:, self.column_indices['y']] == y) & (
                            self.data[:, self.column_indices['elev']] == elev))[0]
                    self.indices_selected.extend(indices)
            self.selection_plot = self.scatter_ax.scatter(self.data[self.indices_selected, self.column_indices['y']],
                                                          self.data[self.indices_selected, self.column_indices['elev']],
                                                          s=1, color='indianred')
            self.show_selected_points_on_map()

    # Label manipulation
    def add_label(self):
        self.last_indices_selected = self.indices_selected
        self.last_edit = 'add'
        self.data[self.indices_selected, self.column_indices['label']] = 1
        self.plot_profile()
        self.edit_flag = True
        # self.selection_polygon.remove()

    def remove_label(self):
        self.last_indices_selected = self.indices_selected
        self.last_edit = 'remove'
        self.data[self.indices_selected, self.column_indices['label']] = 0
        self.plot_profile()
        self.edit_flag = True
        # self.selection_polygon.remove()

    def undo_label(self):
        if self.last_edit == 'add':
            self.data[self.last_indices_selected, self.column_indices['label']] = 0
        else:
            self.data[self.last_indices_selected, self.column_indices['label']] = 1
        self.plot_profile()

    def redo_label(self):
        if self.last_edit == 'add':
            self.data[self.last_indices_selected, self.column_indices['label']] = 1
        else:
            self.data[self.last_indices_selected, self.column_indices['label']] = 0
        self.plot_profile()

    def clear_label(self):
        self.data = self.data_copy
        self.plot_profile()
        # self.selection_polygon.remove()

    # Plot profile
    def plot_profile(self):
        # self.scatter_ax.clear()
        self.figure.clear()
        self.scatter_ax = self.figure.add_subplot(111)

        self.scatter_ax.set_xlabel('Distance (m)', fontsize=8)
        self.scatter_ax.set_ylabel('Elevation (m)', fontsize=8)
        self.scatter_ax.tick_params(axis='both', which='major', labelsize=6)

        y_index = self.column_indices['y']
        elev_index = self.column_indices['elev']

        # # Read column indices
        # try:
        #     y_index = self.column_indices['y']
        #     elev_index = self.column_indices['elev']
        # except:
        #     message_box = QMessageBox()
        #     message_box.setWindowTitle('Error')
        #     message_box.setText('Plot error.')
        #     message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        #     message_box.setIcon(QMessageBox.Icon.Warning)
        #     message_box.exec()
        #     return

        # if self.zoom_x_flag is True:
        #     y_min = self.cur_y_min
        # else:
        #     y_min = self.y_range[0] + self.scroll_bar.value() * self.y_interval
        #     self.cur_y_min = y_min
        # y_max = y_min + self.y_interval

        y_min = self.y_range[0] + self.scroll_bar.value() * self.y_interval/2
        # self.cur_y_min = y_min
        y_max = y_min + self.y_interval

        mask = np.logical_and(self.data[:, y_index] >= y_min, self.data[:, y_index] < y_max)
        y = self.data[mask, y_index]
        elev = self.data[mask, elev_index]

        if self.column_indices['label'] != -1:
            label = self.data[mask, self.column_indices['label']].astype(int)
            if self.column_indices['conf'] != -1:
                conf = self.data[mask, self.column_indices['conf']].astype(int)
                color = np.select([label == 1, (label == 0) & (conf == 1), (label == 0) & (conf == 2)],
                ['orange', 'seagreen', 'royalblue'])
            else:
                color = np.where(label == 0, 'royalblue', 'orange')
            self.scatter_plot = self.scatter_ax.scatter(y, elev, s=1, c=color)
        else:
            self.scatter_plot = self.scatter_ax.scatter(y, elev, s=1)

        # if self.column_indices['label'] != -1:
        #     label = self.data[mask, self.column_indices['label']].astype(int)
        #     if self.column_indices['conf'] != -1:
        #         conf = self.data[mask, self.column_indices['conf']].astype(int)
        #         color = np.select([label == 0, (label == 1) & (conf == 1), (label == 1) & (conf == 2)],
        #           ['royalblue', 'olive', 'olivedrab'])
        #     else:
        #         color = np.where(label == 0, 'royalblue', 'olivedrab')
        #     self.scatter_plot = self.scatter_ax.scatter(y, elev, s=1, c=color)
        # else:
        #     if self.column_indices['conf'] != -1:
        #         conf = self.data[mask, self.column_indices['conf']].astype(int)
        #         color = np.where(conf == 1, 'olive', 'olivedrab')
        #     else:
        #         color = 'royalblue'
        #     self.scatter_plot = self.scatter_ax.scatter(y, elev, s=1, c=color)

        self.scatter_ax.set_xlim(y_min, y_max)

        self.scatter_ax.set_ylim(self.cur_elev_range[0], self.cur_elev_range[1])

        # if self.column_indices['lat'] is not None:
        #     self.scatter_ax2 = self.scatter_ax.twiny()
        #     lat_index = self.column_indices['lat']
        #     # lat = self.data[mask, lat_index]
        #     # self.scatter_ax2.scatter(lat, elev, marker='', alpha=0)
        #     # if lat.size == 0:
        #     index1 = max(np.where(self.data[:, y_index] <= y_min)[0])
        #     index2 = min(np.where(self.data[:, y_index] >= y_max)[0])
        #     lat_min = self.data[index1, lat_index]
        #     lat_max = self.data[index2, lat_index]
        #     self.scatter_ax2.set_xlim(lat_min, lat_max)
        #     # self.scatter_ax2.set_xlim(22, 33)
        # #     else:
        # #         self.scatter_ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{self.lat[int(x)]:.2f}'))
        #     self.scatter_ax2.set_xlabel('Latitude', fontsize=8)
        #     self.scatter_ax2.tick_params(axis='both', which='major', labelsize=6)
        #     self.scatter_ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Adjust the tick formatter for the x-axis
        self.scatter_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        self.figure.tight_layout()

        self.canvas.draw()

    # Plot control
    def on_scroll(self):
        self.plot_profile()

    def get_max_scroll(self):
        return math.ceil((self.y_range[1] - self.y_range[0] - (self.y_interval/2)) / (self.y_interval/2))

    # Import function
    def open_file_dialog(self):
        # Create the file open dialog
        file_dialog = FileDialog(self)

        # Set current directory to the last open directory
        if self.init_dir:
            file_dialog.setDirectory(self.init_dir)

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]

            # Set file tree
            self.root_dir = file_path
            self.init_file_tree()

            # Retrieve current directory
            self.init_dir = file_dialog.directory().path()

            try:
                # Read the file and retrieve the data
                self.basename = os.path.splitext(os.path.basename(file_path))[0]
                self.setWindowTitle(self.basename)
                self.ext = os.path.splitext(os.path.basename(file_path))[1]
                if self.ext == '.txt':
                    df = pd.read_csv(file_path, sep=' ', header=None)
                    self.data = df.to_numpy()
                    self.data_copy = self.data.copy()
                    # self.type = 2
                    # self.column_indices = {
                    #     'lon': 3,
                    #     'lat': 4,
                    #     'y': 1,
                    #     'elev': 2,
                    #     'conf': 6,
                    #     'label': 7
                    # }
                    # Create the preview dialog
                    preview_dialog = PreviewDialog(self.data, self)
                    # Connect the preview dialog's signal to the MainWindow's slot
                    preview_dialog.selection_accepted.connect(self.set_column_indices)
                    # Show the dialog
                    preview_dialog.exec()
                elif self.ext == '.csv':
                    df = pd.read_csv(file_path, sep=',')
                    self.columns = df.columns
                    self.data = df.to_numpy()
                    self.data_copy = self.data.copy()
                    self.column_indices = {
                        'lon':  df.columns.get_loc('lon') if 'lon' in df.columns else -1,
                        'lat':  df.columns.get_loc('lat') if 'lat' in df.columns else -1,
                        'y':  df.columns.get_loc('y') if 'y' in df.columns else -1,
                        'elev':  df.columns.get_loc('elev') if 'elev' in df.columns else -1,
                        'conf': df.columns.get_loc('signal_conf_ph') if 'signal_conf_ph' in df.columns else -1,
                        'label': df.columns.get_loc('annot') if 'annot' in df.columns else (df.columns.get_loc('pred') if 'pred' in df.columns else -1),
                    }
                    # if 'annot' in df.columns:
                    #     self.type = 2
                    # elif 'pred' in df.columns:
                    #     self.type = 1
                    self.set_column_indices()

                # # Create the preview dialog
                # preview_dialog = PreviewDialog(self.data, self)

                # # Connect the preview dialog's signal to the MainWindow's slot
                # preview_dialog.selection_accepted.connect(self.set_column_indices)

                # self.column_indices = {
                #     'lon': None,
                #     'lat': None,
                #     'y': None,
                #     'elev': None,
                #     'label': None
                # }

                # # Show the dialog
                # preview_dialog.exec()
            except:
                message_box = QMessageBox()
                message_box.setWindowTitle('Error')
                message_box.setText('Cannot import file.')
                message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                message_box.setIcon(QMessageBox.Icon.Warning)
                button = message_box.exec()
                if button == QMessageBox.StandardButton.Ok:
                    message_box.close()

    def open_folder_dialog(self):
        # Create the file open dialog
        folder_dialog = FolderDialog(self)

        # Set current directory to the last open directory
        if self.init_dir:
            folder_dialog.setDirectory(self.init_dir)

        self.root_dir = folder_dialog.getExistingDirectory(self, "Select Directory")
        if self.root_dir:
            self.init_file_tree()
            self.init_dir = self.root_dir

    # Export function
    def set_export_folder(self):
        if self.init_dir:
            folder_name = str(os.path.basename(self.init_dir))+'_annotated'
            set_dialog = InputDialog(folder_name)
            if set_dialog.exec() == QDialog.DialogCode.Accepted:
                self.output_dir = os.path.join(os.path.dirname(self.init_dir),folder_name)
                if not os.path.exists(self.output_dir):
                    os.mkdir(self.output_dir)
                    QMessageBox.information(self, "", "Folder '"+folder_name+"' created!")

    def save_file_dialog(self, new=True):
        if self.basename is None:
            return
        else:
            if not self.output_dir:
                QMessageBox.warning(self, "", "No export folder being set!")
                return
            if not new:
                export_file = self.basename
            else:
                export_file = self.basename
            
            export_path = os.path.join(self.output_dir, export_file)

            if self.ext == '.txt':
                selected_filter = "Text Files (*.txt)"
                file_name, selected_filter = QFileDialog.getSaveFileName(self, "Export File", export_path, selected_filter)
                if file_name:
                    df = pd.DataFrame(self.data, columns=self.columns)
                    df.to_csv(file_name, index=False, sep=' ', header=False)
                    # np.savetxt(file_name, self.data, fmt='%.4f', delimiter=' ')
            elif self.ext == '.csv':
                selected_filter = "CSV Files (*.csv)"
                file_name, selected_filter = QFileDialog.getSaveFileName(self, "Export File", export_path, selected_filter)
                if file_name:
                    df = pd.DataFrame(self.data, columns=self.columns)
                    df.to_csv(file_name, index=False)
                    # np.savetxt(file_name, self.data, fmt='%.4f', delimiter=' ')
            
            self.edit_flag = False

    # Open selected file
    def open_selected_file(self, index):
        file = self.model.itemFromIndex(index).text()
        # try:
        #     # Read the file and retrieve the data
        #     self.basename = os.path.splitext(file)[0]
        #     self.setWindowTitle(self.basename)
        #     ext = os.path.splitext(file)[1]
        #     file_fullpath = os.path.join(self.root_dir, file)
        #     if ext == '.txt':
        #         self.data = np.loadtxt(file_fullpath, delimiter=' ')
        #         self.data_copy = np.loadtxt(file_fullpath, delimiter=' ')
        #     elif ext == '.csv':
        #         self.data = np.loadtxt(file_fullpath, delimiter=',', skiprows=1)
        #         self.data_copy = np.loadtxt(file_fullpath, delimiter=',', skiprows=1)

        #     # Create the preview dialog
        #     preview_dialog = PreviewDialog(self.data, self)

        #     # Connect the preview dialog's signal to the MainWindow's slot
        #     preview_dialog.selection_accepted.connect(self.set_column_indices)

        #     # Show the dialog
        #     preview_dialog.exec()
        # except:
        #     message_box = QMessageBox()
        #     message_box.setWindowTitle('Error')
        #     message_box.setText('Cannot import file.')
        #     message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        #     message_box.setIcon(QMessageBox.Icon.Warning)
        #     button = message_box.exec()
        #     if button == QMessageBox.StandardButton.Ok:
        #         message_box.close()      

        try:
            # Read the file and retrieve the data
            self.basename = os.path.splitext(file)[0]
            self.setWindowTitle(self.basename)
            self.ext = os.path.splitext(file)[1]
            file_fullpath = os.path.join(self.root_dir, file)

            if self.ext == '.txt':
                df = pd.read_csv(file_fullpath, sep=' ', header=None)
                self.data = df.to_numpy()
                self.data_copy = self.data.copy()
                # Create the preview dialog
                preview_dialog = PreviewDialog(self.data, self)
                # Connect the preview dialog's signal to the MainWindow's slot
                preview_dialog.selection_accepted.connect(self.set_column_indices)
                # Show the dialog
                preview_dialog.exec()
            elif self.ext == '.csv':
                df = pd.read_csv(file_fullpath, sep=',')
                self.columns = df.columns
                self.data = df.to_numpy()
                self.data_copy = self.data.copy()
                self.column_indices = {
                    'lon':  df.columns.get_loc('lon') if 'lon' in df.columns else -1,
                    'lat':  df.columns.get_loc('lat') if 'lat' in df.columns else -1,
                    'y':  df.columns.get_loc('y') if 'y' in df.columns else -1,
                    'elev':  df.columns.get_loc('elev') if 'elev' in df.columns else -1,
                    'conf': df.columns.get_loc('signal_conf_ph') if 'signal_conf_ph' in df.columns else -1,
                    'label': df.columns.get_loc('annot') if 'annot' in df.columns else (df.columns.get_loc('pred') if 'pred' in df.columns else -1),
                }
                self.set_column_indices()
                
        except:
            message_box = QMessageBox()
            message_box.setWindowTitle('Error')
            message_box.setText('Cannot import file.')
            message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            message_box.setIcon(QMessageBox.Icon.Warning)
            button = message_box.exec()
            if button == QMessageBox.StandardButton.Ok:
                message_box.close()  

    # Set data columns
    def set_column_indices(self, column_indices={}):
        if not self.column_indices:
            self.column_indices = column_indices

        # Read column indices
        if self.column_indices['y'] == -1 or self.column_indices['elev'] == -1:
            message_box = QMessageBox()
            message_box.setWindowTitle('Error')
            message_box.setText('Plot error.')
            message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            message_box.setIcon(QMessageBox.Icon.Warning)
            message_box.exec()
            return
        else:
            self.y_range = (min(self.data[:, self.column_indices['y']]), max(self.data[:, self.column_indices['y']]))
            self.elev_range = (
                min(self.data[:, self.column_indices['elev']]), max(self.data[:, self.column_indices['elev']]))
            self.scroll_bar.setEnabled(True)
            self.scroll_bar.setMaximum(self.get_max_scroll())
            self.cur_elev_range = (self.elev_range[0] - 5, self.elev_range[1] + 5)
            # Plot profile
            self.plot_profile()

        # Display initial map
        self.create_initial_map()

    # Display geographic map
    def create_initial_map(self):
        if self.column_indices['lon'] != -1 and self.column_indices['lat'] != -1:
            self.lat_list = self.data[:, self.column_indices['lat']]
            self.lon_list = self.data[:, self.column_indices['lon']]
            # Compress the coordinates
            coor_cmpr = [(round(lat, 1), round(lon, 1)) for lat, lon in zip(self.lat_list, self.lon_list)]
            # Remove duplicates
            coor_cmpr = [*set(coor_cmpr)]
            self.coor = [{'lat': lat, 'lon': lon} for lat, lon in coor_cmpr]
            self.lat_m = np.mean(self.lat_list)
            self.lon_m = np.mean(self.lon_list)
            self.coor_mean = [self.lat_m, self.lon_m]

            self.coor_selected = []
            self.show_map()

    def show_selected_points_on_map(self):
        selected_lon = self.data[self.indices_selected, self.column_indices['lon']]
        selected_lat = self.data[self.indices_selected, self.column_indices['lat']]

        if len(selected_lon) == 0 or len(selected_lat) == 0:
            self.create_initial_map()
        else:
            lat_m = np.mean(selected_lat)
            lon_m = np.mean(selected_lon)
            self.coor_mean = [lat_m, lon_m]

            coor_cmpr = [(round(lat, 2), round(lon, 2)) for lat, lon in zip(selected_lat, selected_lon)]
            # Remove duplicates
            coor_cmpr = [*set(coor_cmpr)]
            self.coor_selected = [{'lat': lat, 'lon': lon} for lat, lon in coor_cmpr]

            self.show_map()

    def show_map(self):
        template = env.get_template('map.html')
        # pass data to js file
        html = template.render(coor=self.coor_mean, data=self.coor, sData=self.coor_selected)
        if self.browser is not None:
            self.browser.setHtml(html)
            self.browser.show()

    # Open attribution window
    def open_attr_window(self):
        self.window = AttributionWindow()
        self.window.show()

    # Close window
    def closeEvent(self, event):
        if len(self.column_indices) != 0:
            if self.edit_flag:
                reply = QMessageBox.question(self, 'Close window', 'You have unsaved changes, '
                                                                   'are you sure you want to exit?',
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
            else:
                reply = QMessageBox.question(self, 'Close window', 'Are you sure you want to exit?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                event.accept()  # Close the main window
            else:
                event.ignore()  # Keep the main window open


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AntWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())