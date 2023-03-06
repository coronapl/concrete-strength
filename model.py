# Pablo Valencia A01700912
# March 6, 2023
# Intelligent Systems

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


FEATURES = ['cement', 'blast_furnace_slag', 'fly_ash', 'water']

def load_dataset(filepath, columns):
    df = pd.read_excel(filepath)
    df.columns = columns
    return df

def create_model(x_train, y_train):
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    return model

def get_model_performance(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)
    coefficient_determination = r2_score(test_data, predictions)
    return mse, coefficient_determination

def train_model():
    print('-- CONCRETE STRENGTH PREDICTOR --\n')
    print('Training the model...')

    df = load_dataset('dataset/Concrete_Data.xls', [
                'cement',
                'blast_furnace_slag', 
                'fly_ash', 'water',
                'superplasticizer',
                'coarse_aggregate',
                'fine_aggregate',
                'age',
                'compressive_strength'])

    df_x = df[FEATURES]
    df_y = df['compressive_strength']

    x_train, x_test = df_x[:950], df_x[950:]
    y_train, y_test = df_y[:950], df_y[950:]

    model = create_model(x_train, y_train)
    y_predictions = model.predict(x_test)
    mse, coefficient_determination = get_model_performance(y_test, y_predictions)
   
    print('-----------------------------------\n')
    print('Model performance:\n')
    print('Coefficients: \n', model.coef_)
    print('Mean squared error: %.2f' % mse)
    print('Coefficient of determination: %.2f' % coefficient_determination)

    return model

def get_user_data():
    print('\nUser input: \n')

    cement = float(input('Kg of cement in m^3 of mixture: ')) 
    blast_furnace_slag = float(input('Kg of blast furnace slag in m^3 of mixture: ')) 
    fly_ash = float(input('Kg of fly ash in m^3 of mixture: ')) 
    water = float(input('Kg of water in m^3 of mixture: ')) 

    return [cement, blast_furnace_slag, fly_ash, water]

def main():
    model = train_model()
    stop = False

    while not stop:
        user_data = get_user_data()
        user_df = pd.DataFrame([user_data], columns=FEATURES)
        
        print('The predicted concrete strength is %.2f MPa\n' % 
              model.predict(user_df))

        stop_input = input('Do you want to make another prediction? yes/no\n')
        stop = True if stop_input == 'no' else False

main()

