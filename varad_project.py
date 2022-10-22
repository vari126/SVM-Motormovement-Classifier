from scipy.io import loadmat as lm
import numpy as np
from scipy import signal
import math
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

file_train="C:\\Users\Varad\\Desktop\\EEG-SVM\\Competition_train.mat"
train_dict={}
train_raw=lm(file_train,train_dict)
x_train=train_dict['X']
y_train=train_dict['Y']
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)

#Defining number of EEG trials , channels, sampling frequency and time
trials=278
electrode_channels=64
time_series=3000
sampling_frequency=100
time=3000/sampling_frequency

#Setting alpha, beta, gamma and delta frequency window
alpha_min=8
alpha_max=15
beta_min=16
beta_max=31
theta_min=4
theta_max=7
delta_min=0.5
delta_max=4

#defining sample window for FFT decompisition
win=4*sampling_frequency

#Initialising variables
alpha_features=0
beta_features=0
delta_features=0
theta_features=0

alpha=[None]
beta=[None]
theta=[None]
delta=[None]
features=np.array([])

#Calculating the RMS average of powerspectrum across 4 frequency bands
for trial in range(trials):
     for channel in range(electrode_channels):

          freqs,psd=signal.welch(x_train[trial,channel,:],sampling_frequency,nperseg=win)

          beta = psd[np.logical_and(freqs >= beta_min, freqs <= beta_max)]
          beta = [i ** 2 for i in beta]
          beta_features = sum(beta)
          beta_features = math.sqrt(beta_features)
          features=np.append(features,beta_features)

          alpha=psd[np.logical_and(freqs>=alpha_min,freqs<= alpha_max)]
          alpha =[i ** 2 for i in alpha]
          alpha_features=sum(alpha)
          alpha_features=math.sqrt(alpha_features)
          features = np.append(features, alpha_features)

          theta =psd[np.logical_and(freqs >= theta_min,freqs <= theta_max)]
          theta = [i ** 2 for i in theta]
          theta_features = sum(theta)
          theta_features = math.sqrt(theta_features)
          features = np.append(features, theta_features)

          delta= psd[np.logical_and(freqs >= delta_min,freqs <= delta_max)]
          delta=[i ** 2 for i in delta]
          delta_features=sum(delta)
          delta_features=math.sqrt(theta_features)
          features=np.append(features,delta_features)



features=np.reshape(features,(-1,64*4))
x_train, x_test, y_train, y_test = train_test_split(features, y_train, test_size=0.33, random_state=42)
y_train=np.ravel(y_train)

#polynomial fit
clf_poly=svm.SVC(kernel='poly',degree=3, gamma =10)
svm_model_poly=clf_poly.fit(x_train,y_train)
prediction_poly=svm_model_poly.predict(x_test)
print("POLYNOMIAL FIT /n")
print(metrics.accuracy_score(y_test,prediction_poly))
print(metrics.classification_report(y_test,prediction_poly))

#linear fit
clf_linear=svm.SVC(kernel='linear')
svm_model_linear=clf_linear.fit(x_train,y_train)
prediction_linear=svm_model_linear.predict(x_test)

print("LINEAR FIT /n")
print(metrics.accuracy_score(y_test,prediction_linear))
print(metrics.classification_report(y_test,prediction_linear))

#rbf fit
clf_rbf=svm.SVC(kernel='rbf')
svm_model_rbf=clf_rbf.fit(x_train,y_train)
prediction_rbf=svm_model_rbf.predict(x_test)

print("RBF FIT /n")
print(metrics.accuracy_score(y_test,prediction_rbf))
print(metrics.classification_report(y_test,prediction_rbf))
