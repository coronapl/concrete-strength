import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def load_dataset(filepath, columns):
    df = pd.read_excel(filepath)
    df.columns = columns
    return df

def scale_dataframe(df):
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df

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
    scaled_df = scale_dataframe(df)

    df_x = scaled_df[['cement', 'blast_furnace_slag', 'fly_ash', 'water']]
    df_y = scaled_df['compressive_strength']

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

def main():
    model = train_model()

    print('\n User input: \n')

    # Get user input
    cement_kg = int(input('Kg of cement in m^3 of mixture: ')) 
    blast_furnace_slag_kg = int(input('Kg of blast furnace slag in m^3 of mixture: ')) 
    fly_ash_kg = int(input('Kg of fly ash in m^3 of mixture: ')) 
    water_kg = int(input('Kg of water in m^3 of mixture: ')) 

    user_data = [cement_kg, blast_furnace_slag_kg, fly_ash_kg, water_kg]
    user_data = pd.DataFrame([user_data], columns=['cement', 'blast_furnace_slag', 'fly_ash', 'water'])
    scaled_user_data = scale_dataframe(user_data)
    
    print(model.predict(scaled_user_data))

main()
