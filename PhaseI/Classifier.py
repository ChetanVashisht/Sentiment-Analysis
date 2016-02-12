from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import csv
import numpy as np
from os import listdir

def decode(j, flag):
    if flag == 1:
        if(j == 'S'):
            return "Sadness"
        elif(j == 'H'):
            return "Happiness"
        elif(j == 'F'):
            return "Fear"
        elif(j == 'A'):
            return "Anger"
        elif(j == 'N'):
            return "Neutal"
    else:
        if(j == 0):
            return "Sadness"
        elif(j == 1):
            return "Happiness"
        elif(j == 2):
            return "Fear"
        elif(j == 3):
            return "Anger"
        elif(j == 4):
            return "Neutal"

def extractor(file_input = 'Features.csv'):
    f = open(file_input, 'rb')
    name = []
    reader = csv.reader(f)
    label = []
    for row in reader:
            name.append(row[0])
            label.append(row[len(row) - 1])

    data = np.genfromtxt("Features.csv", delimiter = ',')
    data = np.delete(data, 0, 1)
    data = np.delete(data, len(data[1])-1, 1)
    
    # { Sadness :0, Happiness:1, Fear:2, Anger:3, Neutral:4 }
    for i, j in enumerate(label):
        if(j == 'S'):
            label[i] = 0
        elif(j == 'H'):
            label[i] = 1
        elif(j == 'F'):
            label[i] = 2
        elif(j == 'A'):
            label[i] = 3
        elif(j == 'N'):
            label[i] = 4
    
    labels = zip(name, label)
    return data, labels


def documenter(names, results_boosting, results_Bayes, results_NN, results_KNN, accuracy_Boosting, accuracy_Bayes, accuracy_NN, accuracy_KNN):
    # Documenting the results
    with open('results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['File', 'Expected Emotion', 'Boosting', 'Baye\'s Classifier', 'Neural Net', 'KNN Classifier'])
        for i in xrange(len(names)):
            writer.writerow([names[i], decode(names[i][-5], 1), decode(results_boosting[i], 0), decode(results_Bayes[i], 0), decode(results_NN[i], 0), decode(results_KNN[i], 0)])
        writer.writerow(['% Accuracy', 100, accuracy_Boosting*100, accuracy_Bayes*100, accuracy_NN*100, accuracy_KNN*100])


def trainer(dataset = "Features.csv"):
    # Train the various machine learning algorithms using the features extracted.
    data, labels = extractor(dataset)
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size = 0.20, random_state = 42)
    names, expected_results = zip(*test_labels)
    names1, train_labels = zip(*train_labels)
    
    print 'S' + '\t' + 'H' + '\t' + 'F' + '\t' + 'A' + '\t' + 'N'
    
    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators = 100, n_jobs = 2)
    rf.fit(train, train_labels)
    results_boosting = rf.predict(test)
    conf_matrix = confusion_matrix(expected_results, results_boosting)
    print "Forset Classifier:\n"
    print conf_matrix
    accuracy_Boosting = float(np.trace(conf_matrix))/float(np.sum(conf_matrix))
    print accuracy_Boosting

    # KNN Classifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train, train_labels)
    results_KNN = neigh.predict(test)
    conf_matrix = confusion_matrix(expected_results, results_KNN)
    print "KNN Classifier:\n"
    print conf_matrix
    accuracy_KNN = float(np.trace(conf_matrix))/float(np.sum(conf_matrix))
    print accuracy_KNN

    # Baye's Classifier
    clf = GaussianNB()
    clf.fit(train, train_labels)
    results_Bayes = clf.predict(test)
    conf_matrix = confusion_matrix(expected_results, results_Bayes)
    print "\nBayes Classifier:\n"
    print conf_matrix
    accuracy_Bayes = float(np.trace(conf_matrix))/float(np.sum(conf_matrix))
    print accuracy_Bayes

    # Neural Network
    clf = BernoulliNB()
    clf.fit(train, train_labels)
    results_NN = clf.predict(test)
    conf_matrix = confusion_matrix(expected_results, results_NN)
    print "\nNeural Network:\n"
    print conf_matrix
    accuracy_NN = float(np.trace(conf_matrix))/float(np.sum(conf_matrix))
    print accuracy_NN

    documenter(names, results_boosting, results_Bayes, results_NN, results_KNN, accuracy_Boosting, accuracy_Bayes, accuracy_NN, accuracy_KNN)

