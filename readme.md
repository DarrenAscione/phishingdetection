# Machine Learning Classifier for Identification of Phishing Attacks based on URL Analysis

[![Build Status](https://travis-ci.org/DarrenAscione/phishingdetection.svg?branch=master)](https://travis-ci.org/DarrenAscione/phishingdetection)

## Project Structure

```
.
├── readme.md
├── data
|   ├── dataset.csv
|   ├── tree.dot
|   └── features.md
└── main
    ├── helper_class.py
    ├── decision_tree.py
    ├── svm.py
    └── neural_net.py
```

## Introduction

## Abstract

Phishing is defined as the attempt to obtain sensitive and private information such as usernames, passwords, and credit card details, often for malicious reasons, by disguising as a trustworthy entity through the internet. In the recent years, more than 4,000 ransomware attacks have occurred every day since the beginning of 2016. In perspective, the rate of ransomware attacks on businesses is at an alarming rate of once every 40 seconds. The main reason for this increase is through the use of phishing. Phishing emails and pop-up websites have continued to grow as an attack vector for ransomware. It is therefore of paramount importance that we have a capable means of identifying such malicious sites to prevent cybercrimes. Phishing can be seen as a binary classification problem in data mining where the classifier is constructed using large number of website’s features such as URL and IP addresses.

There are many forms of phishing such as spear phishing, clone phishing, whaling and filter evasion. This project focuses on the aspect of link manipulation of phishing where some form of technical deception has been designed to appear to belong to a spoofed organisation. Phishers have taken advantage of URL redirectors on websites of trusted organisations to disguise malicious URLs with a trusted domain.

## Introduction

Phishing as defined according to (Abdelhamid, et al., 2014) as the art of mimicking a legitimate website in order to deceive users by obtaining their personal private information such as usernames, passwords, account numbers etc. The most common mean of initiating a phishing attack is through email where a link to a fake website will be attached to the email. The fake website may look almost identical to the user and hence once the user logins with his/her password, it will be redirected to the attacker.

Most of the countermeasures available today against such phishing websites endeavour to identify the legitimacy of these websites through techniques such as warning the user that a particular website maybe a phish. But these warning tools are generated through simple rule-based system. As such, as the number of phishing sites increases, these anti-phishing tools cannot keep up as it will have keep adding new rules for every additional new phishing site that is created. Thus, there is a serious scaling issue at hand with the current approach of identifying phishing websites through such rule-based systems.

## Proposed Solution

The method proposed in this project addresses this lacuna by suggesting a better way of identifying such phishing sites through the exploration and use of machine learning algorithms.  
The purpose of this project is to apply various machine learning techniques to quickly identify these malicious sites and to find structures and patterns in the URL of these sites. This can help organisations and individuals to safely and accurately flag a phishing site and to collectively build a database of phishing sites so as to protect both individuals and organisations from such cyber crimes.

These machine learning algorithms will be applied to solve this binary classification problem if a website is phished or not. The following algorithms were explored, tested and analysed for their efficiency and accuracy on the dataset collected. Furthermore, optimisation techniques and data processing methods have also been explored to improve the efficiency of these algorithms.

Machine Learning Algorithms that were explored:

- Decision Tree  
- Neural Network 
- Support Vector Machine 
- Logistic Regression


