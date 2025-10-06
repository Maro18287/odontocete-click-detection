# odontocete-click-detection
This challenge specifically aims to detect the presence of odontoceti clicks in underwater audio recordings in the Caribbean sea. The model will be evaluated on the Challenge Data ENS website.

## Data Description
The dataset is composed of 23,168 audio files in WAV format, each of duration 200ms. The clicks are labeled with a binary variable: 1 if the file contains a click, 0 otherwise.

## Challenge
The objective of the challenge is to create a model that predicts the presence of odontoceti clicks in the test set with the highest accuracy.

## Evaluation
The submissions are evaluated on the ROC AUC (area under the curve) metric.

The results must be submitted as a CSV file with 950 lines. Each line corresponds to a file of the test set and contains the prediction for this file. The prediction in percentage should be indicated and must not be rounded to binary labels.
