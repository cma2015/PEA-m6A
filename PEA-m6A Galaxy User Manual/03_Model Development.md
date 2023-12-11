# <center>PEA-m6A User Manual (version 1.0)</center>

- PEA-m6A is an ensemble learning framework for predicting m6A modifications at regional-scale.
- PEA-m6A consists of four modules:**Sample Preparation, Feature Encoding, Model Development and Model Assessment**, each of which contains a comprehensive collection of functions with pre-specified parameters available.
- PEA-m6A was powered with an advanced  packaging technology, which enables compatibility and portability.
- PEA-m6A project is hosted on http://github.com/cma2015/PEA-m6A
- PEA-m6A docker image is available at http://hub.docker.com/r/malab/peam6a
- PEA-m6A server can be accessed via http://peam6a.omstudio.cloud

## PEA-m6A Model Development

This module contains the **Prediction System Constrction** used to construct an RNA modification predictor at region-scale and provides predictive function among 12 plant species.

| Functions                         | **Description**                                              | **Input**                                           | **Output**                               | **Reference**    |
| --------------------------------- | ------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------- | ---------------- |
| **Prediction System Constrction** | construct an RNA modification predictor at region-scale and provides predictive function among 12 plant species. | Positive feature matrix and negative feature matrix | A predictor and model evaluation results | In-house scripts |

## **Prediction System Constrction** 

This function contains the **Prediction System Constrction** used to construct an RNA modification predictor at region-scale and provides predictive function among 12 plant species.

#### Input

- **Feature matrix of positive samples**
- **Feature matrix of negative samples**

#### Output

- An RNA modification predictor in binary format

![prediction_system](/Users/smg/Desktop/user manual/img/prediction_system.png)
