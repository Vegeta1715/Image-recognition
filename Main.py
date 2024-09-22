from sklearn.datasets import fetch_olivetti_faces
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
#importing image database
image_data = fetch_olivetti_faces()

#separating features,target from dataset
features = image_data.data

#print(features.shape) -> 400 rows(40persons*10images), 4096 columns -> Features
target = image_data.target
#print(target.shape) ->400 rows 1 column representing person number


#creating subplots to see unique person's image
# fig, sub_plot = plt.subplots(nrows=8, ncols=5, figsize=(15, 20))
# sub_plot = sub_plot.flatten()
# for unique_p in np.unique(target):
#     index = unique_p*8
#     sub_plot[unique_p].imshow(features[index].reshape(64, 64), cmap='gray')
#     sub_plot[unique_p].set_xticks([])
#     sub_plot[unique_p].set_yticks([])
#     sub_plot[unique_p].set_title("Person ID = %s" % unique_p)

#showing 1 person's 10 photos
# fig, sub_plot = plt.subplots(nrows=1, ncols=10, figsize=(10, 20))
# sub_plot = sub_plot.flatten()
# for person in range(0, 10):
#     sub_plot[person].imshow(features[person].reshape(64, 64), cmap='gray')
#     sub_plot[person].set_xticks([])
#     sub_plot[person].set_yticks([])
#     sub_plot[person].set_title("Person ID = 0")


#spilitting into training, test set
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, stratify=target, random_state=0)
#setting accuracy 95% and applying pca to get lesser number of features from which we can get same result
pca = PCA(0.95)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
x_pca = pca.transform(features)
#print(len(pca.components_)) to see number of eigen vectors
# eigen_faces = pca.components_.reshape(len(pca.components_), 64, 64)
# fig, sub_plot = plt.subplots(nrows=10, ncols=12, figsize=(17, 15))
# sub_plot=sub_plot.flatten()
# for i in range(len(pca.components_)):
#     sub_plot[i].imshow(eigen_faces[i], cmap='gray')
#     sub_plot[i].set_xticks([])
#     sub_plot[i].set_yticks([])
# plt.suptitle('Eigen faces')
# plt.show()

models = [("Logistic Regression", LogisticRegression()), ("SVM", SVC()), ("Naive Bayes classifier", GaussianNB())]
for name, model in models:
    k = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, x_pca, target, cv=k)
    print("Mean of score for %s is %s" % (name, cv_scores.mean()))
    # ML_model = model
    # ML_model.fit(x_train, y_train)
    # predicted = ML_model.predict(x_test)
    # print("Results with %s :" % name)
    # print("Accuracy: %s" % metrics.accuracy_score(y_test,predicted))

