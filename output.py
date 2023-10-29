import os
import sys
import re
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtCore import QDir
from OutputWindow import Ui_OutputWindow

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_beam_id(file):
    basename = os.path.splitext(os.path.basename(file))[0]
    beam_id = basename.split('_')[-5][-2:]
    return beam_id

def get_beam_id(filename):
    pattern = r'gt[1-3][lr]'
    match = re.search(pattern, filename)
    
    if match:
        beam_id = match.group(0)
        return beam_id
    else:
        return ''

def refraction_correction_approx(b_z, w_z):
    b_z = b_z + 0.25416 * (w_z - b_z)
    return b_z

def refraction_correction(df):
    if 'pred' in df.columns:
        sf_class = 'pred'
    else:
        sf_class = 'annotation'
    df_sf = df.loc[df[sf_class] != 0]
    if df_sf.empty:
        pass
    else:
        # find water surface photons
        df_w = df.loc[(df[sf_class] == 0) & (df["class"] == 5)]

        # Maximum distance to consider for nearby water surface points
        max_distance = 100  # Adjust as needed

        # Create relative water surface elevation column
        df['w_elev'] = None

        # Iterate through seafloor points and apply correction in the original DataFrame 'df'
        for index, sf_point in df_sf.iterrows():
            # Find all water surface points within the maximum distance
            nearby_water_surface_points = df_w[abs(df_w['y'] - sf_point['y']) <= max_distance]
            
            if not nearby_water_surface_points.empty:
                # Calculate the mean elevation of the nearby water surface points
                mean_elevation = nearby_water_surface_points['elev'].mean()
                
                # Apply refraction correction based on the mean elevation
                corrected_elevation = refraction_correction_approx(sf_point['elev'], mean_elevation)
            else:
                # No nearby water surface points found, no correction applied
                mean_elevation = None
                corrected_elevation = sf_point['elev']
            
            # Update the 'elev' value in the original DataFrame 'df'
            df.at[index, 'elev'] = corrected_elevation
            df.at[index, 'w_elev'] = mean_elevation

        # # obtain original seafloor elevation and coordinates
        # sf_y = df_sf['y'].to_numpy()
        # min_y = np.min(sf_y)
        # max_y = np.max(sf_y)

        # # obtain the water surface level by section (500m)
        # df_corrected_sf = df_sf.copy()
        # num = math.ceil((max_y - min_y) / 500)
        # y1 = min_y
        # for i in range(num):
        #     y2 = y1 + 500
        #     w_elev = np.mean(df_w.loc[(df_w['y'] >= y1) & (df_w['y'] < y2), ['elev']].to_numpy())
        #     sf_elev = df_sf.loc[(df['y'] >= y1) & (df['y'] < y2), ['elev']].to_numpy()
        #     sf_elev = refraction_correction_approx(sf_elev, w_elev)
        #     df_corrected_sf.loc[(df['y'] >= y1) & (df['y'] < y2), ['elev']] = sf_elev
        #     y1 = y2

        # df.loc[df[sf_class] != 0, 'elev'] = df_corrected_sf['elev']

    return df

# class FileDialog(QFileDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Open File")
#         self.setFileMode(QFileDialog.FileMode.ExistingFile)
#         self.setNameFilters(["CSV Files (*.csv)"])
#         self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)

class FolderDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Folder")
        self.setFileMode(QFileDialog.FileMode.Directory)

class OutputWindow(QMainWindow, Ui_OutputWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.input_dir = None
        self.root_dir = None
        self.dir_basename = None
        self.merge_dir = None
        self.output_dir = None
        self.columns = None

        self.inputPathBtn.clicked.connect(self.inputFolderDialog)
        self.outputPathButton.clicked.connect(self.setExportFolder)
        self.outputBtn.clicked.connect(self.output_process)

    def inputFolderDialog(self):
        folder_dialog = FolderDialog(self)

        self.input_dir = folder_dialog.getExistingDirectory(self, "Select Input Path")

        self.inputPath.setPlainText(self.input_dir)

        self.root_dir = os.path.dirname(self.input_dir)
        self.dir_basename = os.path.basename(self.input_dir)

    def setExportFolder(self):
        if self.outputPath.toPlainText() and self.root_dir:
            self.output_dir = os.path.join(self.root_dir, self.outputPath.toPlainText())
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
                QMessageBox.information(self, "", "Folder '"+self.outputPath.toPlainText()+"' created!")

    def output_process(self):
        self.showStartPopup()
        try:
            self.columns = pd.read_csv(os.path.join(self.input_dir,os.listdir(self.input_dir)[0])).columns
            self.merge()
            self.post_process()
            self.showDonePopup('Done')
        except:
            self.showDonePopup('Failed')

    def merge(self):
        self.merge_dir = os.path.join(self.root_dir,self.dir_basename+'_merge')
        if not os.path.exists(self.merge_dir):
            os.mkdir(self.merge_dir)

        file_list = []
        pattern = r'^(.*[NS])'
        for sub_file in os.listdir(self.input_dir):
            match = re.search(pattern, sub_file)
            if match:
                file_list.append(match.group(0))

        file_list = set(file_list)

        for file in file_list:
            sub_file_list = []
            for sub_file in os.listdir(self.input_dir):
                if file in sub_file:
                    df_sub_file = pd.read_csv(os.path.join(self.input_dir, sub_file), sep=',')
                    sub_file_list.extend(df_sub_file.to_numpy().tolist())
            df = pd.DataFrame(sub_file_list, columns=self.columns)
            # convert label column to integer
            if 'pred' in df.columns:
                df['pred'] = df['pred'].astype(int)
            if 'label' in df.columns:
                df['label'] = df['label'].astype(int)
            output_file = os.path.join(self.merge_dir, file + '.csv')
            df.to_csv(output_file, sep=',', index=False, header=True)

    def post_process(self):
        if not self.output_dir:
            self.output_dir = os.path.join(self.root_dir, self.dir_basename+'_output')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
        
        # columns = ["x", "y", "lon", "lat", "elev", "signal_conf_ph", "class", "annotation"]
        # columns = ['x', 'y', 'elev', 'lon', 'lat', 'class', 'prob', 'pred']
        # df_first = pd.read_csv(os.listdir(self.input_dir)[0])
        df_all = pd.DataFrame(columns=self.columns)
        df_all['beam'] = None
        if self.merge_dir:
            list_dir = self.merge_dir
            filename_first = os.path.splitext(os.listdir(self.merge_dir)[0])[0]
            filename_first = filename_first[:-13]
        else:
            list_dir = self.input_dir
            filename_first = os.path.splitext(os.listdir(self.input_dir)[0])[0]
            filename_first = filename_first[:-16]
        for basefile in os.listdir(list_dir):
            filename = os.path.splitext(basefile)[0]
            file = os.path.join(list_dir, basefile)
            df = pd.read_csv(file)

            # refraction correction
            if self.rcorrBox.isChecked():
                df = refraction_correction(df)

            # keep only bathymetric photons
            if self.sfRadioBtn.isChecked():
                if 'annotation' in self.columns:
                    df= df[df['annotation'] == 1]
                if 'pred' in self.columns:
                    df = df[df['pred']==1]

            # combine all tracks
            if self.oneTrackRadioBtn.isChecked():
                # length = len(df_all)
                # concatenate to new df
                if not df.empty:
                    df_all = pd.concat([df_all, df], ignore_index=True)
                    # retrieve the beam id
                    beam_id = get_beam_id(file)

                    # record the beam id
                    df_all.loc[df_all.index[-len(df):], 'beam'] = beam_id
                    # df_all.loc[length:length+len(df), 'beam'] = beam_id
            else:
                if not df.empty:
                    # output seperate tracks to a folder
                    if self.txtRadioBtn.isChecked():
                        output_file = os.path.join(self.output_dir, filename+'.txt')
                    else:
                        output_file = os.path.join(self.output_dir, filename+'.csv')
                    df.to_csv(output_file, index=False)

        if not df_all.empty:
            if self.oneTrackRadioBtn.isChecked():
                # output combined file
                if self.txtRadioBtn.isChecked():
                    output_file = os.path.join(self.output_dir, filename_first+'.txt')
                else:
                    output_file = os.path.join(self.output_dir, filename_first+'.csv')
                df_all.to_csv(output_file, index=False)

    def batch_process(self):
        for file in os.listdir(self.dir):
            refraction_correction(file)

    def showStartPopup(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Starting...")
        msg_box.setText("The operation has started.")
        msg_box.exec()

    def showDonePopup(self, result):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(result)
        msg_box.setText(result+'!')
        msg_box.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OutputWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())