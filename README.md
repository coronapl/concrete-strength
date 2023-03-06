# Concrete Strength Predictor Using Linear Regression

This model is a simple linear regression that uses the dataset "Concrete Compressive Strength" by Prof. I-Cheng Yeh, which was obtained from the UCI Machine Learning Repository.

I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)

## Problem

According to the additional information from the dataset, the compressive strength of concrete is a highly non-linear function of age and ingredients. Measuring the compressive strength of concrete is done in laboratories with specialized equipment by breaking cylindrical concrete specimens with a machine while measuring the load used.

## Solution

This model was trained to predict the compressive strength of concrete by considering the amount of cement, blast furnace slag, fly ash, and water in the mixture. The goal is to create a predictor that can be used to optimize the strength of the concrete and reduce material waste.

The model's coefficients are:

[0.11099095 0.08263888 0.07164461 -0.16514799]

Mean squared error: 51.14
Coefficient of determination: 0.68

The coefficient of determination of the model is 0.68, which is a good result considering the complexity of predicting the compressive strength of concrete since it is a non-linear function of age and ingredients. It's also important to note that the dataset is quite small, so the model may be too simple for the value we are trying to predict. Having a larger dataset could improve the performance of the model.

## Usage

Clone the repository from GitHub (coronapl/concrete-strength).
Create a virtual environment with the command python3 -m venv venv.
Activate the virtual environment with the command source venv/bin/activate.
Install all required packages with the command pip3 install -r requirements.txt.
Run the program with the command python3 model.py.
Enter the kg of cement in m³ of mixture (100 - 550).
Enter the kg of blast furnace slag in m³ of mixture (0 - 360).
Enter the kg of fly ash in m³ of mixture (0 - 200).
Enter the kg of water in m³ of mixture (100 - 250).
