# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI2.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import sys, os, cv2, xlwt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from Recognition import PlateRecognition


class Ui_MainWindow(object):

    def __init__(self):
        self.RowLength = 0
        self.Data = [['文件名称', '录入时间', '车牌号码', '车牌类型', '识别耗时', '车牌信息']]

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1213, 670)
        MainWindow.setFixedSize(1213, 670)  # 设置窗体固定大小
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(690, 10, 511, 491))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 509, 489))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label_0 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_0.setGeometry(QtCore.QRect(10, 10, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_0.setFont(font)
        self.label_0.setObjectName("label_0")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label.setGeometry(QtCore.QRect(10, 40, 481, 441))
        self.label.setObjectName("label")
        self.label.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea_2 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_2.setGeometry(QtCore.QRect(10, 10, 671, 631))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_1 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_1.setGeometry(QtCore.QRect(0, 0, 669, 629))
        self.scrollAreaWidgetContents_1.setObjectName("scrollAreaWidgetContents_1")
        self.label_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_1)
        self.label_1.setGeometry(QtCore.QRect(10, 10, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.tableWidget = QtWidgets.QTableWidget(self.scrollAreaWidgetContents_1)
        self.tableWidget.setGeometry(QtCore.QRect(10, 40, 651, 581))  # 581))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setColumnWidth(0, 140)  # 设置1列的宽度
        self.tableWidget.setColumnWidth(1, 130)  # 设置2列的宽度
        self.tableWidget.setColumnWidth(2, 65)  # 设置3列的宽度
        self.tableWidget.setColumnWidth(3, 75)  # 设置4列的宽度
        self.tableWidget.setColumnWidth(4, 65)  # 设置5列的宽度
        self.tableWidget.setColumnWidth(5, 174)  # 设置6列的宽度

        self.tableWidget.setHorizontalHeaderLabels(["图片名称", "录入时间", "识别耗时", "车牌号码", "车牌类型", "车牌信息"])
        self.tableWidget.setRowCount(self.RowLength)
        self.tableWidget.verticalHeader().setVisible(False)  # 隐藏垂直表头)
        # self.tableWidget.setStyleSheet("selection-background-color:blue")
        # self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.raise_()
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_1)
        self.scrollArea_3 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_3.setGeometry(QtCore.QRect(690, 510, 341, 131))
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 339, 129))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.label_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_3.setGeometry(QtCore.QRect(10, 40, 321, 81))
        self.label_3.setObjectName("label_3")
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.scrollArea_4 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_4.setGeometry(QtCore.QRect(1040, 510, 161, 131))
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollArea_4.setObjectName("scrollArea_4")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 159, 129))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_4)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 50, 121, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents_4)
        self.pushButton.setGeometry(QtCore.QRect(20, 90, 121, 31))
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.__openimage)  # 设置点击事件
        self.pushButton_2.clicked.connect(self.__writeFiles)  # 设置点击事件
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.ProjectPath = os.getcwd()  # 获取当前工程文件位置

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车牌识别系统"))
        self.label_0.setText(_translate("MainWindow", "原始图片："))
        self.label.setText(_translate("MainWindow", ""))
        self.label_1.setText(_translate("MainWindow", "识别结果："))
        self.label_2.setText(_translate("MainWindow", "车牌区域："))
        self.label_3.setText(_translate("MainWindow", ""))
        self.pushButton.setText(_translate("MainWindow", "打开文件"))
        self.pushButton_2.setText(_translate("MainWindow", "导出数据"))
        self.label_4.setText(_translate("MainWindow", "命令："))
        self.scrollAreaWidgetContents_1.show()

    # 识别
    def __vlpr(self, path):
        PR = PlateRecognition()
        result = PR.VLPR(path)
        return result

    def __show(self, result, FileName):
        # 显示表格
        self.RowLength = self.RowLength + 1
        if self.RowLength > 18:
            self.tableWidget.setColumnWidth(5, 157)
        self.tableWidget.setRowCount(self.RowLength)
        self.tableWidget.setItem(self.RowLength - 1, 0, QTableWidgetItem(FileName))
        self.tableWidget.setItem(self.RowLength - 1, 1, QTableWidgetItem(result['InputTime']))
        self.tableWidget.setItem(self.RowLength - 1, 2, QTableWidgetItem(str(result['UseTime']) + '秒'))
        self.tableWidget.setItem(self.RowLength - 1, 3, QTableWidgetItem(result['Number']))
        self.tableWidget.setItem(self.RowLength - 1, 4, QTableWidgetItem(result['Type']))
        if result['Type'] == '蓝色牌照':
            self.tableWidget.item(self.RowLength - 1, 4).setBackground(QBrush(QColor(3, 128, 255)))
        elif result['Type'] == '绿色牌照':
            self.tableWidget.item(self.RowLength - 1, 4).setBackground(QBrush(QColor(98, 198, 148)))
        elif result['Type'] == '黄色牌照':
            self.tableWidget.item(self.RowLength - 1, 4).setBackground(QBrush(QColor(242, 202, 9)))
        self.tableWidget.setItem(self.RowLength - 1, 5, QTableWidgetItem(result['From']))

        # 显示识别到的车牌位置
        size = (int(self.label_3.width()), int(self.label_3.height()))
        shrink = cv2.resize(result['Picture'], size, interpolation=cv2.INTER_AREA)
        shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink[:], shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def __writexls(self, DATA, path):
        wb = xlwt.Workbook();
        ws = wb.add_sheet('Data');
        # DATA.insert(0, ['文件名称','录入时间', '车牌号码', '车牌类型', '识别耗时', '车牌信息'])
        for i, Data in enumerate(DATA):
            for j, data in enumerate(Data):
                ws.write(i, j, data)
        wb.save(path)
        QMessageBox.information(None, "成功", "数据已保存！", QMessageBox.Yes)

    def __writecsv(self, DATA, path):
        f = open(path, 'w')
        # DATA.insert(0, ['文件名称','录入时间', '车牌号码', '车牌类型', '识别耗时', '车牌信息'])
        for data in DATA:
            f.write((',').join(data) + '\n')
        f.close()
        QMessageBox.information(None, "成功", "数据已保存！", QMessageBox.Yes)

    def __writeFiles(self):
        path, filetype = QFileDialog.getSaveFileName(None, "另存为", self.ProjectPath,
                                                     "Excel 工作簿(*.xls);;CSV (逗号分隔)(*.csv)")
        if path == "":  # 未选择
            return
        if filetype == 'Excel 工作簿(*.xls)':
            self.__writexls(self.Data, path)
        elif filetype == 'CSV (逗号分隔)(*.csv)':
            self.__writecsv(self.Data, path)

    def __openimage(self):
        path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                     "JPEG Image (*.jpg);;PNG Image (*.png);;JFIF Image (*.jfif)")  # ;;All Files (*)
        if path == "":  # 未选择文件
            return
        filename = path.split('/')[-1]

        # 尺寸适配
        size = cv2.cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR).shape
        if size[0] / size[1] > 1.0907:
            w = size[1] * self.label.height() / size[0]
            h = self.label.height()
            jpg = QtGui.QPixmap(path).scaled(w, h)
        elif size[0] / size[1] < 1.0907:
            w = self.label.width()
            h = size[0] * self.label.width() / size[1]
            jpg = QtGui.QPixmap(path).scaled(w, h)
        else:
            jpg = QtGui.QPixmap(path).scaled(self.label.width(), self.label.height())

        self.label.setPixmap(jpg)
        result = self.__vlpr(path)
        if result is not None:
            self.Data.append(
                [filename, result['InputTime'], result['Number'], result['Type'], str(result['UseTime']) + '秒',
                 result['From']])
            self.__show(result, filename)
        else:
            QMessageBox.warning(None, "Error", "无法识别此图像！", QMessageBox.Yes)


# 重写MainWindow类
class MainWindow(QtWidgets.QMainWindow):

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, '提示',
                                               "是否要退出程序？\n提示：退出后将丢失所有识别数据",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    if os.path.exists('provinces.json'):
        if os.path.exists('cardtype.json'):
            if os.path.exists('Prefecture.json'):
                if os.path.exists('config.js'):
                    app = QtWidgets.QApplication(sys.argv)
                    MainWindow = MainWindow()  # QtWidgets.QMainWindow()
                    ui = Ui_MainWindow()
                    ui.setupUi(MainWindow)
                    MainWindow.show()
                    sys.exit(app.exec_())
                else:
                    print('未找到参数文件 config.js')
                    RuntimeError('未找到参数文件 config.js')
            else:
                print('未找到 Prefecture.json 文件')
                RuntimeError('未找到 Prefecture.json 文件')
        else:
            print('未找到 cardtype.json 文件')
            RuntimeError('未找到 cardtype.json 文件')
    else:
        print('未找到 provinces.json 文件')
        RuntimeError('未找到 provinces.json 文件')
