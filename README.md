# **NLP4CyberSecurity**


[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/jackaduma/CycleGAN-VC2)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/jackaduma?locale.x=zh_XC)

[**中文说明**](./README.zh-CN.md) | [**English**](./README.md)

------

This code is  NLP  models and tech  implementation for **cyber security** task, driven by deep learning model, a nice work on **cyber security**.

本项目使用自然语言处理（NLP）技术应用于网络安全领域，包括恶意软件检测、漏洞发现和威胁情报等方面。该项目基于Python编程语言和机器学习框架Scikit-learn、TensorFlow和Keras等，实现了一些常见的NLP技术，如文本预处理、特征提取、词嵌入、文本分类和主题建模等。通过对网络安全方面的文本数据进行处理和分析，该项目能够提高网络安全人员的工作效率和准确性，以及更好地发现网络安全威胁。此外，该项目还提供了一些用于网络安全的NLP数据集和预训练模型，方便其他研究人员和开发者使用。

- [x] Dataset
  - [x] weak password
  - [x] xss injection
  - [x] malicious url
  - [x] phishing url
- [x] Usage
  - [x] Training
  - [x] Example 
- [ ] Demo
- [x] Reference

------

## **Update**

------


## **Table of Contents**

- [**NLP4CyberSecurity**](#nlp4cybersecurity)
  - [**Update**](#update)
  - [**Table of Contents**](#table-of-contents)
  - [**Requirement**](#requirement)
  - [**Usage**](#usage)
  - [**Weak Password Detection**](#weak-password-detection)
    - [**Eval Result**](#eval-result)
  - [**XSS Injection Detection**](#xss-injection-detection)
    - [**simple nn model**](#simple-nn-model)
    - [**simple cnn model**](#simple-cnn-model)
    - [**simple lstm model**](#simple-lstm-model)
  - [**Malicious URL Detection**](#malicious-url-detection)
    - [**RNN**](#rnn)
    - [**CNN**](#cnn)
    - [**Conv LSTM**](#conv-lstm)
  - [**Phishing URL Detection**](#phishing-url-detection)
  - [**Demo**](#demo)
  - [**Star-History**](#star-history)
  - [**Reference**](#reference)
  - [**Donation**](#donation)
  - [**License**](#license)
  
------



## **Requirement** 

```bash
pip install -r requirements.txt
```
## **Usage**


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

------


## **Demo**

Samples:

```
```

------
## **Star-History**

![star-history](https://api.star-history.com/svg?repos=jackaduma/NLP4CyberSecurity&type=Date "star-history")

------

## **Reference**

------

## **Donation**
If this project help you reduce time to develop, you can give me a cup of coffee :) 

AliPay(支付宝)
<div align="center">
	<img src="./misc/ali_pay.png" alt="ali_pay" width="400" />
</div>

WechatPay(微信)
<div align="center">
    <img src="./misc/wechat_pay.png" alt="wechat_pay" width="400" />
</div>

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://paypal.me/jackaduma?locale.x=zh_XC)

------

## **License**

[MIT](LICENSE) © Kun
