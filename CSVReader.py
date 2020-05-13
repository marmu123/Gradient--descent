import csv
import numpy as np
class CSVReader:
    def __init__(self,filename):
        self.filename=filename
        self.data={}
        self.columnNames=[]
        self.noRows=0
        self.trainData={}
        self.testData={}

        self.readData()
        self.splitData()


    def readData(self):
        csvdata=[]
        with open(self.filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            self.noRows = 0
            for row in csv_reader:
                if self.noRows == 0:
                    self.columnNames = row
                else:
                    csvdata.append(row)
                self.noRows += 1
        self.noRows-=1
        for i,key in enumerate(self.columnNames):
            self.data[key]=[el[i] for el in csvdata]

    def splitData(self):
        listData=[]
        for i in range(self.noRows):
            el=[]
            for key in self.data.keys():
                #print(key,i)
                el.append(self.data[key][i])
            listData.append(el)

        #np.random.seed(5)
        indexes = [i for i in range(self.noRows)]
        trainIndexes = np.random.choice(indexes, int(0.8 * self.noRows), replace=False)
        testIndexex = [i for i in indexes if i not in trainIndexes]

        trainData = [listData[i] for i in trainIndexes]
        testData = [listData[i] for i in testIndexex]


        for i,key in enumerate(self.columnNames):
            self.trainData[key] = [el[i] for el in trainData]
            self.testData[key] = [el[i] for el in testData]

    def scaleMinMax(self,features):
        minFeat = min(features)
        maxFeat = max(features)
        scaledFeatures = [(float(feat) - float(minFeat)) / (float(maxFeat) - float(minFeat)) for feat in features]
        return scaledFeatures







