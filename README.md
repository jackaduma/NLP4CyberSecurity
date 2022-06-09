# **NLP4CyberSecurity**

NLP  model and tech  for cyber security task







---


## [**Weak Password Detection**](./01_weak_password_detect.ipynb)

weak password detection with machine learning

weak-password/password-strength detection with machine learning; 弱密码检测；密码强度检测

### **Eval Result**

```

              precision    recall  f1-score   support

           0    0.94406   0.83240   0.88472      8920
           1    0.96327   0.98971   0.97631     49652
           2    0.99035   0.95400   0.97184      8392

    accuracy                        0.96428     66964
   macro avg    0.96589   0.92537   0.94429     66964
weighted avg    0.96410   0.96428   0.96355     66964

```


---

## [**XSS Injection Detection**](02_xss_injection_detect.ipynb)

xss injection detection with machine learning

### **simple nn model**

```
Precision score is : 0.9764296754250387
Recall score is : 0.9830772223302859
```

### **simple cnn model**

```
Precision score is : 0.9948463825569871
Recall score is : 0.9762692083252286
```

### **simple lstm model**

```
Precision score is : 0.9980311084859225
Recall score is : 0.9869548286604362
```
---

## [**Malicious URL Detection**](03_malicious_url_detect.ipynb)

malicious url detection with machine learning

### **RNN**

```
Accuracy Score is:  0.8655441478439425
Precision Score is : 0.8579050828418984
Recall Score is : 0.8767578205075642
F1 Score:  0.8672290036092299
AUC Score:  0.8655252346603806
```

### **CNN**

```
Accuracy Score is:  0.8379671457905544
Precision Score is : 0.8431494883953082
Recall Score is : 0.831085236357673
F1 Score:  0.8370738958974254
AUC Score:  0.8379787529437384
```


### **Conv LSTM**

```
Accuracy Score is:  0.9242505133470226
Precision Score is : 0.9288969917958068
Recall Score is : 0.9191095076052642
F1 Score:  0.92397733127254
AUC Score:  0.9242591842604873
```

---

## [**Phishing URL Detection**](04_phishing_url_detect.ipynb)

phishing url detection with machine learning

```
accuracy: 0.9982
Model Accuracy: 99.82%
```

```
              precision    recall  f1-score   support

           0    0.99790   0.99895   0.99843      1904
           1    0.99866   0.99732   0.99799      1495

    accuracy                        0.99823      3399
   macro avg    0.99828   0.99814   0.99821      3399
weighted avg    0.99824   0.99823   0.99823      3399

```