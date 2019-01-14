from PyQt5 import QtCore, QtGui, QtWidgets

import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            line = f.read()
            lines += line.split('\n')
            f.close()
            message = lines
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)
 
data = DataFrame ({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory(r'C:\Users\Albertus Heronius\Documents\opinion\ask', 'ask'),sort=True)
data = data.append(dataFrameFromDirectory(r'C:\Users\Albertus Heronius\Documents\opinion\compliment', 'compliment'),sort=True)
data = data.append(dataFrameFromDirectory(r'C:\Users\Albertus Heronius\Documents\opinion\hate', 'hate'),sort=True)
data = data.append(dataFrameFromDirectory(r'C:\Users\Albertus Heronius\Documents\opinion\request', 'request'),sort=True)
data = data.append(dataFrameFromDirectory(r'C:\Users\Albertus Heronius\Documents\opinion\spam', 'spam'),sort=True)

data['message']=[" ".join(message) for message in data['message'].values]
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

class Ui_main(object):
    def setupUi(self, main):
        main.setObjectName("main")
        main.resize(659, 535)
        self.labelInput = QtWidgets.QLabel(main)
        self.labelInput.setGeometry(QtCore.QRect(10, 10, 111, 19))
        self.labelInput.setObjectName("labelInput")
        self.labelResult = QtWidgets.QLabel(main)
        self.labelResult.setGeometry(QtCore.QRect(480, 10, 68, 19))
        self.labelResult.setObjectName("labelResult")
        self.startBtn = QtWidgets.QPushButton(main)
        self.startBtn.setGeometry(QtCore.QRect(550, 490, 101, 34))
        self.startBtn.setObjectName("startBtn")
        self.addBtn = QtWidgets.QPushButton(main)
        self.addBtn.setGeometry(QtCore.QRect(570, 450, 81, 34))
        self.addBtn.setObjectName("addBtn")
        self.clearAllBtn = QtWidgets.QPushButton(main)
        self.clearAllBtn.setGeometry(QtCore.QRect(430, 490, 112, 34))
        self.clearAllBtn.setObjectName("clearAllBtn")
        self.listResult = QtWidgets.QListWidget(main)
        self.listResult.setGeometry(QtCore.QRect(480, 40, 171, 401))
        self.listResult.setObjectName("listResult")
        self.lineEdit = QtWidgets.QLineEdit(main)
        self.lineEdit.setGeometry(QtCore.QRect(10, 450, 551, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.listInput = QtWidgets.QListWidget(main)
        self.listInput.setGeometry(QtCore.QRect(10, 40, 461, 401))
        self.listInput.setObjectName("listInput")

        self.retranslateUi(main)
        QtCore.QMetaObject.connectSlotsByName(main)

        self.addBtn.clicked.connect(self.addItem)

        self.startBtn.clicked.connect(self.startML)

        self.clearAllBtn.clicked.connect(self.clearAll)

    def retranslateUi(self, main):
        _translate = QtCore.QCoreApplication.translate
        main.setWindowTitle(_translate("main", "Public Opinion Filter"))
        self.labelInput.setText(_translate("main", "Input data"))
        self.labelResult.setText(_translate("main", "Result"))
        self.startBtn.setText(_translate("main", "Start"))
        self.addBtn.setText(_translate("main", "Add"))
        self.clearAllBtn.setText(_translate("main", "Clear All"))

    def addItem(self):
        value = self.lineEdit.text()
        self.lineEdit.clear()
        self.listInput.addItem(value)

    def startML(self):
        self.listResult.clear()
        list=[]
        count = self.listInput.count()
        if (count==0):
            pass
        else:
            for i in range(0, count):
                list.append(self.listInput.item(i).text())
            example_count = vectorizer.transform(list)
            predictions = classifier.predict(example_count)
            for i in range(0, count):
                self.listResult.addItem(predictions[i])

    def clearAll(self):
        self.listResult.clear()
        self.listInput.clear()
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = QtWidgets.QDialog()
    ui = Ui_main()
    ui.setupUi(main)
    main.show()
    sys.exit(app.exec_())