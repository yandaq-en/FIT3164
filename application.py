import sys
from PyQt5 import QtWidgets
import pandas as pd
import xgboost as xgb
import numpy as np


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.frame = pd.DataFrame()
        self.init_ui()

    def init_ui(self):
        # define all buttons and labels in the left
        self.b_reset = QtWidgets.QPushButton("Reset")
        self.b_predict = QtWidgets.QPushButton("Predict")
        self.b_clear = QtWidgets.QPushButton("Clear")
        self.l_age = QtWidgets.QLabel("Age")
        self.l_gender = QtWidgets.QLabel("Gender(M/F)")
        self.l_height = QtWidgets.QLabel("Height")
        self.l_icu = QtWidgets.QLabel("ICU Type(1,2,3,4)")
        self.l_alp = QtWidgets.QLabel("ALP")
        self.l_alt = QtWidgets.QLabel("ALT")
        self.l_ast = QtWidgets.QLabel("AST")
        self.l_albumin = QtWidgets.QLabel("Albumin")
        self.l_bun = QtWidgets.QLabel("BUN")
        self.l_bilirubin = QtWidgets.QLabel("Bilirubin")
        self.l_cholesterol = QtWidgets.QLabel("Cholesterol")
        self.l_creatinine = QtWidgets.QLabel("Creatinine")
        self.l_diasabp = QtWidgets.QLabel("DiasABP")
        self.l_fio2 = QtWidgets.QLabel("FiO2")
        self.l_gcs = QtWidgets.QLabel("GCS")
        self.l_glucose = QtWidgets.QLabel("Glucose")
        self.l_hco3 = QtWidgets.QLabel("HCO3")
        self.l_hct = QtWidgets.QLabel("HCT")
        self.l_hr = QtWidgets.QLabel("HR")
        self.l_k = QtWidgets.QLabel("K")
        self.l_lactate = QtWidgets.QLabel("Lactate")
        self.l_map = QtWidgets.QLabel("MAP")
        self.l_mg = QtWidgets.QLabel("Mg")
        self.l_nidiasabp = QtWidgets.QLabel("NIDiasABP")
        self.l_nimap = QtWidgets.QLabel("NIMAP")
        self.l_nisysabp = QtWidgets.QLabel("NISysABP")
        self.l_na = QtWidgets.QLabel("Na")
        self.l_paco2 = QtWidgets.QLabel("PaCO2")
        self.l_pao2 = QtWidgets.QLabel("PaO2")
        self.l_platelets = QtWidgets.QLabel("Platelets")
        self.l_resprate = QtWidgets.QLabel("RespRate")
        self.l_sao2 = QtWidgets.QLabel("SaO2")
        self.l_sysabp = QtWidgets.QLabel("SysABP")
        self.l_temp = QtWidgets.QLabel("Temp")
        self.l_troponinI = QtWidgets.QLabel("TroponinI")
        self.l_troponinT = QtWidgets.QLabel("TroponinT")
        self.l_urine = QtWidgets.QLabel("Urine")
        self.l_wbc = QtWidgets.QLabel("WBC")
        self.l_weight = QtWidgets.QLabel("Weight")
        self.l_ph = QtWidgets.QLabel("PH")

        # define all text box in the left
        self.t_age = QtWidgets.QLineEdit()
        self.t_gender = QtWidgets.QLineEdit()
        self.t_height = QtWidgets.QLineEdit()
        self.t_icu = QtWidgets.QLineEdit()
        self.t_alp = QtWidgets.QLineEdit()
        self.t_alt = QtWidgets.QLineEdit()
        self.t_ast = QtWidgets.QLineEdit()
        self.t_albumin = QtWidgets.QLineEdit()
        self.t_bun = QtWidgets.QLineEdit()
        self.t_bilirubin = QtWidgets.QLineEdit()
        self.t_cholesterol = QtWidgets.QLineEdit()
        self.t_creatinine = QtWidgets.QLineEdit()
        self.t_diasabp = QtWidgets.QLineEdit()
        self.t_fio2 = QtWidgets.QLineEdit()
        self.t_gcs = QtWidgets.QLineEdit()
        self.t_glucose = QtWidgets.QLineEdit()
        self.t_hco3 = QtWidgets.QLineEdit()
        self.t_hct = QtWidgets.QLineEdit()
        self.t_hr = QtWidgets.QLineEdit()
        self.t_k = QtWidgets.QLineEdit()
        self.t_lactate = QtWidgets.QLineEdit()
        self.t_map = QtWidgets.QLineEdit()
        self.t_mg = QtWidgets.QLineEdit()
        self.t_nidiasabp = QtWidgets.QLineEdit()
        self.t_nimap = QtWidgets.QLineEdit()
        self.t_nisysabp = QtWidgets.QLineEdit()
        self.t_na = QtWidgets.QLineEdit()
        self.t_paco2 = QtWidgets.QLineEdit()
        self.t_pao2 = QtWidgets.QLineEdit()
        self.t_platelets = QtWidgets.QLineEdit()
        self.t_resprate = QtWidgets.QLineEdit()
        self.t_sao2 = QtWidgets.QLineEdit()
        self.t_sysabp = QtWidgets.QLineEdit()
        self.t_temp = QtWidgets.QLineEdit()
        self.t_troponinI = QtWidgets.QLineEdit()
        self.t_troponinT = QtWidgets.QLineEdit()
        self.t_urine = QtWidgets.QLineEdit()
        self.t_wbc = QtWidgets.QLineEdit()
        self.t_weight = QtWidgets.QLineEdit()
        self.t_ph = QtWidgets.QLineEdit()

        # define all boxes
        self.whole = QtWidgets.QHBoxLayout()
        self.right = QtWidgets.QVBoxLayout()
        self.left = QtWidgets.QVBoxLayout()
        self.button = QtWidgets.QHBoxLayout()
        self.static = QtWidgets.QHBoxLayout()
        self.temporal = QtWidgets.QVBoxLayout()
        self.temporal_1 = QtWidgets.QHBoxLayout()
        self.temporal_2 = QtWidgets.QHBoxLayout()
        self.temporal_3 = QtWidgets.QHBoxLayout()
        self.temporal_4 = QtWidgets.QHBoxLayout()
        self.temporal_5 = QtWidgets.QHBoxLayout()
        self.temporal_6 = QtWidgets.QHBoxLayout()
        self.temporal_7 = QtWidgets.QHBoxLayout()
        self.temporal_8 = QtWidgets.QHBoxLayout()
        self.temporal_9 = QtWidgets.QHBoxLayout()

        # complete box static
        self.static.addWidget(self.l_age)
        self.static.addWidget(self.t_age)
        self.static.addWidget(self.l_gender)
        self.static.addWidget(self.t_gender)
        self.static.addWidget(self.l_height)
        self.static.addWidget(self.t_height)
        self.static.addWidget(self.l_icu)
        self.static.addWidget(self.t_icu)

        # complete box temporal_1
        self.temporal_1.addWidget(self.l_alp)
        self.temporal_1.addWidget(self.t_alp)
        self.temporal_1.addWidget(self.l_alt)
        self.temporal_1.addWidget(self.t_alt)
        self.temporal_1.addWidget(self.l_ast)
        self.temporal_1.addWidget(self.t_ast)
        self.temporal_1.addWidget(self.l_albumin)
        self.temporal_1.addWidget(self.t_albumin)

        # complete box temporal_2
        self.temporal_2.addWidget(self.l_bun)
        self.temporal_2.addWidget(self.t_bun)
        self.temporal_2.addWidget(self.l_bilirubin)
        self.temporal_2.addWidget(self.t_bilirubin)
        self.temporal_2.addWidget(self.l_cholesterol)
        self.temporal_2.addWidget(self.t_cholesterol)
        self.temporal_2.addWidget(self.l_creatinine)
        self.temporal_2.addWidget(self.t_creatinine)

        # complete box temporal_3
        self.temporal_3.addWidget(self.l_diasabp)
        self.temporal_3.addWidget(self.t_diasabp)
        self.temporal_3.addWidget(self.l_fio2)
        self.temporal_3.addWidget(self.t_fio2)
        self.temporal_3.addWidget(self.l_gcs)
        self.temporal_3.addWidget(self.t_gcs)
        self.temporal_3.addWidget(self.l_glucose)
        self.temporal_3.addWidget(self.t_glucose)

        # complete box temporal_4
        self.temporal_4.addWidget(self.l_hco3)
        self.temporal_4.addWidget(self.t_hco3)
        self.temporal_4.addWidget(self.l_hct)
        self.temporal_4.addWidget(self.t_hct)
        self.temporal_4.addWidget(self.l_hr)
        self.temporal_4.addWidget(self.t_hr)
        self.temporal_4.addWidget(self.l_k)
        self.temporal_4.addWidget(self.t_k)

        # complete box temporal_5
        self.temporal_5.addWidget(self.l_lactate)
        self.temporal_5.addWidget(self.t_lactate)
        self.temporal_5.addWidget(self.l_map)
        self.temporal_5.addWidget(self.t_map)
        self.temporal_5.addWidget(self.l_mg)
        self.temporal_5.addWidget(self.t_mg)
        self.temporal_5.addWidget(self.l_nidiasabp)
        self.temporal_5.addWidget(self.t_nidiasabp)

        # complete box temporal_6
        self.temporal_6.addWidget(self.l_nimap)
        self.temporal_6.addWidget(self.t_nimap)
        self.temporal_6.addWidget(self.l_nisysabp)
        self.temporal_6.addWidget(self.t_nisysabp)
        self.temporal_6.addWidget(self.l_na)
        self.temporal_6.addWidget(self.t_na)
        self.temporal_6.addWidget(self.l_paco2)
        self.temporal_6.addWidget(self.t_paco2)

        # complete box temporal_7
        self.temporal_7.addWidget(self.l_pao2)
        self.temporal_7.addWidget(self.t_pao2)
        self.temporal_7.addWidget(self.l_platelets)
        self.temporal_7.addWidget(self.t_platelets)
        self.temporal_7.addWidget(self.l_resprate)
        self.temporal_7.addWidget(self.t_resprate)
        self.temporal_7.addWidget(self.l_sao2)
        self.temporal_7.addWidget(self.t_sao2)

        # complete box temporal_8
        self.temporal_8.addWidget(self.l_sysabp)
        self.temporal_8.addWidget(self.t_sysabp)
        self.temporal_8.addWidget(self.l_temp)
        self.temporal_8.addWidget(self.t_temp)
        self.temporal_8.addWidget(self.l_troponinI)
        self.temporal_8.addWidget(self.t_troponinI)
        self.temporal_8.addWidget(self.l_troponinT)
        self.temporal_8.addWidget(self.t_troponinT)

        # complete box temporal_9
        self.temporal_9.addWidget(self.l_urine)
        self.temporal_9.addWidget(self.t_urine)
        self.temporal_9.addWidget(self.l_wbc)
        self.temporal_9.addWidget(self.t_wbc)
        self.temporal_9.addWidget(self.l_weight)
        self.temporal_9.addWidget(self.t_weight)
        self.temporal_9.addWidget(self.l_ph)
        self.temporal_9.addWidget(self.t_ph)

        # complete box temporal
        self.temporal.addLayout(self.temporal_1)
        self.temporal.addLayout(self.temporal_2)
        self.temporal.addLayout(self.temporal_3)
        self.temporal.addLayout(self.temporal_4)
        self.temporal.addLayout(self.temporal_5)
        self.temporal.addLayout(self.temporal_6)
        self.temporal.addLayout(self.temporal_7)
        self.temporal.addLayout(self.temporal_8)
        self.temporal.addLayout(self.temporal_9)

        # complete box button
        self.button.addWidget(self.b_predict)
        self.button.addWidget(self.b_clear)
        self.button.addWidget(self.b_reset)

        # complete box left
        self.left.addLayout(self.static)
        self.left.addLayout(self.temporal)
        self.left.addLayout(self.button)

        # complete box right
        self.result = QtWidgets.QLabel("The probability of surviving will be shown here")
        self.right.addWidget(self.result)

        # complete box whole and set layout
        self.whole.addLayout(self.left)
        self.whole.addLayout(self.right)
        self.setLayout(self.whole)

        # set he title of the window
        self.setWindowTitle("ICU Mortality Prediction")

        self.b_predict.clicked.connect(self.predict)
        self.b_clear.clicked.connect(self.clear)
        self.b_reset.clicked.connect(self.reset)
        self.show()

    def reset(self):
        self.t_alp.setText("")
        self.t_alt.setText("")
        self.t_ast.setText("")
        self.t_albumin.setText("")
        self.t_bun.setText("")
        self.t_bilirubin.setText("")
        self.t_cholesterol.setText("")
        self.t_creatinine.setText("")
        self.t_diasabp.setText("")
        self.t_fio2.setText("")
        self.t_gcs.setText("")
        self.t_glucose.setText("")
        self.t_hco3.setText("")
        self.t_hct.setText("")
        self.t_hr.setText("")
        self.t_k.setText("")
        self.t_lactate.setText("")
        self.t_map.setText("")
        self.t_mg.setText("")
        self.t_nidiasabp.setText("")
        self.t_nimap.setText("")
        self.t_nisysabp.setText("")
        self.t_na.setText("")
        self.t_paco2.setText("")
        self.t_pao2.setText("")
        self.t_platelets.setText("")
        self.t_resprate.setText("")
        self.t_sao2.setText("")
        self.t_sysabp.setText("")
        self.t_temp.setText("")
        self.t_troponinI.setText("")
        self.t_troponinT.setText("")
        self.t_urine = QtWidgets.QLineEdit()
        self.t_wbc.setText("")
        self.t_weight.setText("")
        self.t_ph.setText("")
        self.t_age.setEnabled(True)
        self.t_gender.setEnabled(True)
        self.t_height.setEnabled(True)
        self.t_icu.setEnabled(True)
        self.t_age.setText("")
        self.t_gender.setText("")
        self.t_height.setText("")
        self.t_icu.setText("")
        self.frame = pd.DataFrame()

    def clear(self):
        self.t_alp.setText("")
        self.t_alt.setText("")
        self.t_ast.setText("")
        self.t_albumin.setText("")
        self.t_bun.setText("")
        self.t_bilirubin.setText("")
        self.t_cholesterol.setText("")
        self.t_creatinine.setText("")
        self.t_diasabp.setText("")
        self.t_fio2.setText("")
        self.t_gcs.setText("")
        self.t_glucose.setText("")
        self.t_hco3.setText("")
        self.t_hct.setText("")
        self.t_hr.setText("")
        self.t_k.setText("")
        self.t_lactate.setText("")
        self.t_map.setText("")
        self.t_mg.setText("")
        self.t_nidiasabp.setText("")
        self.t_nimap.setText("")
        self.t_nisysabp.setText("")
        self.t_na.setText("")
        self.t_paco2.setText("")
        self.t_pao2.setText("")
        self.t_platelets.setText("")
        self.t_resprate.setText("")
        self.t_sao2.setText("")
        self.t_sysabp.setText("")
        self.t_temp.setText("")
        self.t_troponinI.setText("")
        self.t_troponinT.setText("")
        self.t_urine = QtWidgets.QLineEdit()
        self.t_wbc.setText("")
        self.t_weight.setText("")
        self.t_ph.setText("")

    def predict(self):
        self.t_age.setDisabled(True)
        self.t_gender.setDisabled(True)
        self.t_height.setDisabled(True)
        self.t_icu.setDisabled(True)

        newrow = [self.t_alp.text(), self.t_alt.text(), self.t_ast.text(), self.t_albumin.text(),
                  self.t_bun.text(), self.t_bilirubin.text(), self.t_cholesterol.text(), self.t_creatinine.text(),
                  self.t_diasabp.text(), self.t_fio2.text(), self.t_gcs.text(), self.t_glucose.text(),
                  self.t_hco3.text(), self.t_hct.text(), self.t_hr.text(), self.t_k.text(), self.t_lactate.text(),
                  self.t_map.text(), self.t_mg.text(), self.t_nidiasabp.text(), self.t_nimap.text(),
                  self.t_nisysabp.text(),
                  self.t_na.text(), self.t_paco2.text(), self.t_pao2.text(), self.t_platelets.text(),
                  self.t_resprate.text(), self.t_sao2.text(), self.t_sysabp.text(), self.t_temp.text(),
                  self.t_troponinI.text(), self.t_troponinT.text(), self.t_urine.text(), self.t_wbc.text(),
                  self.t_weight.text(), self.t_ph.text()]
        self.frame = self.frame.append(pd.Series(newrow, index=['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
                                                                'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS',
                                                                'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP',
                                                                'Mg',
                                                                'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2',
                                                                'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp',
                                                                'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight',
                                                                'pH']), ignore_index=True)
        self.frame = self.frame.replace("", np.NaN)
        self.frame = self.frame.apply(pd.to_numeric)
        self.static = pd.DataFrame(data=[self.t_age.text(), int(self.t_gender.text() == 'M'), self.t_height.text(),
                                         0, 0, 0],
                                   index=['Age', 'Gender', 'Height', 'ICUType_2.0', 'ICUType_3.0', 'ICUType_4.0']).T
        self.static = self.static.apply(pd.to_numeric)

        if self.t_icu.text() != "":
            for i in range(2, 5):
                if int(self.t_icu.text()) == i:
                    self.static['ICUType_' + str(i)] = 1
        self.static = self.static.replace('', np.NaN)
        self.mean = pd.DataFrame(self.frame.mean()).T
        self.min = pd.DataFrame(self.frame.min()).T
        self.max = pd.DataFrame(self.frame.max()).T
        self.mean = self.mean.add_suffix('_mean')
        self.min = self.min.add_suffix('_min')
        self.max = self.max.add_suffix('_max')
        self.evaluate = pd.concat([self.static, self.mean, self.min, self.max], axis=1)
        self.evaluate = self.evaluate.reindex(sorted(self.evaluate.columns), axis=1)
        self.evaluate = self.evaluate[list(train_X)]
        for i in range(len(list(train_X))):
            if np.isnan(self.evaluate.iloc[0, i]):
                self.evaluate.iloc[0, i] = impute[i]
        yhat = model.predict_proba(self.evaluate)[0][0]
        self.result.setText("Probapility of surviving is " + str(yhat))


df = open('param.txt', 'r')
content = df.read().replace('{', '')
content = content.replace('}', '')
content = content.split(',')
for i in range(len(content)):
    content[i] = content[i].split(':')
content = dict(content)
df.close()

train_X = pd.read_csv('selected_train_X.csv')
impute = list(train_X.mean())
train_y = pd.read_csv('train_y.csv')['In-hospital_death']
model = xgb.XGBClassifier(gamma=float(content["'gamma'"]), learning_rate=float(content[" 'learning_rate'"]),
                          max_delta_step=float(content[" 'max_delta_step'"]),
                          max_depth=int(float(content[" 'max_depth'"])),
                          min_child_weight=float(content[" 'min_child_weight'"]),
                          n_estimators=int(float(content[" 'n_estimators'"])),
                          reg_alpha=float(content[" 'reg_alpha'"]), reg_lambda=float(content[" 'reg_lambda'"]),
                          scale_pos_weight=float(content[" 'scale_pos_weight'"]),
                          subsample=float(content[" 'subsample'"]))
model.fit(train_X, train_y)

app = QtWidgets.QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())