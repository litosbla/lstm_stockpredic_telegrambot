
"""
Codigo creado por litosbla 
https://github.com/litosbla

"""
import json
import telebot
import yfinance as yf
import mplfinance as mpf
import time
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, TimeDistributed, Attention, Reshape
from numpy import polyfit
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.callbacks import Callback
class CustomEarlyStopping(Callback):
    def __init__(self, monitor='val_loss', threshold=4.5e-4):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is not None and current_loss < self.threshold:
            print(f"\nDeteniendo el entrenamiento temprano en la época {epoch+1} debido a {self.monitor} alcanzado el umbral de {self.threshold}.")
            self.model.stop_training = True

# Token del bot
with open("config.json","r") as file:
    tk=json.load(file)
TOKEN = tk["telegram_token"]
bot = telebot.TeleBot(TOKEN)


# Funcion donde todo va a ser genial
def toma_datos(symbol):
    mt5.initialize()
    # Obtener los precios históricos de EUR/USD en un timeframe de 4 horas
    if symbol == '':
        symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    days_to_subtract = 1000
    current_date = datetime.now()
    date_minus_1217_days = current_date - timedelta(days=days_to_subtract)
    formatted_date = date_minus_1217_days.strftime("%Y-%m-%d")
    start_time = pd.Timestamp(formatted_date)
    current_time = pd.Timestamp.now()  # Obtiene la fecha y hora actual
    end_time = current_time + pd.Timedelta(hours=10) 
    symbol_prices = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    # Convertir los precios en un DataFrame de pandas
    df = pd.DataFrame(symbol_prices)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def create_sequences(data, seq_length, future_steps):
    x, y = [], []
    for i in range(len(data) - seq_length - future_steps + 1):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+future_steps].reshape(-1)) 
    return np.array(x), np.array(y)
    
def trainer(data_scaled, seq_lenght, future_steps, features):
    x_train, y_train = create_sequences(data_scaled, seq_lenght, future_steps)
    
    inputs = Input(shape=(seq_lenght, len(features)))
    lstm = LSTM(128, return_sequences=True)(inputs)
    attention = Attention()([lstm, lstm])
    dropout = Dropout(0.2)(attention)
    lstm2 = LSTM(64)(dropout)
    output = Dense(future_steps*len(features))(lstm2)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    #custom_early_stop = CustomEarlyStopping()
    custom_early_stopping = CustomEarlyStopping(monitor='val_loss', threshold=4.5e-4)
   
    #history = model.fit(x_train, y_train, epochs=20, batch_size=23, validation_split=0.1)
    history = model.fit(x_train, y_train, epochs=20, batch_size=23, callbacks=[custom_early_stopping], validation_split=0.1)
    model.save("modelo_entrenado.h5")
    return model

def calculate_slopes(data):
    num_series = data.shape[1]
    slopes = []

    for i in range(num_series):
        for j in range(i + 1, num_series):
            series_1 = data[:, i]
            series_2 = data[:, j]

            model = LinearRegression()
            model.fit(series_1.reshape(-1, 1), series_2)
            
            slope = model.coef_[0]
            slopes.append((i, j, slope))
    
    return slopes

def calculate_general_slope(data):
    tiempo = np.arange(0, data.shape[0])
    pendientes = []

    for i in range(data.shape[1]):
        slope, _ = np.polyfit(tiempo, data[:, i], 1)
        pendientes.append(slope)

    pendiente_general = np.mean(pendientes)
    return pendiente_general


def entrenamiento(rentrenar): 
    if rentrenar:
        try:
            os.remove(r'modelo_entrenado.h5')
        except OSError as e:
            print("no se pudo remover modelo para reentrenar o no hay ningun modelo")
    t_1 = time.time()
    seq_lenght=500
    future_steps=5
    features=['close','high','low']
    data=toma_datos('EURUSD')[features]
    print(data[:-1])
    data=data[:-1]
    
    scaler=MinMaxScaler()
    data_scaled=scaler.fit_transform(data)
    print(data_scaled)
    data_array=scaler.inverse_transform(data_scaled)
    if os.path.exists(r'modelo_entrenado.h5'):
        model=load_model(r'modelo_entrenado.h5')
        
    else:
        model=trainer(data_scaled, seq_lenght, future_steps,features)
        
    input_data = data_scaled[-seq_lenght:].reshape(1,-1,len(features)) #ok
    input_data_A = data_scaled[-(seq_lenght+1):-1].reshape(1,-1,len(features))
    predictions = model.predict(input_data)
    predictions_A=model.predict(input_data_A)
    # print(data_scaled.shape)
    # print(predictions.shape)
    # print(predictions_A.shape)
    # nueva_matriz = np.split(predictions, [3], axis=1)
    # nueva_matriz_A=np.split(predictions_A,[3],axis=1)
    # print(nueva_matriz.shape)
    # print(nueva_matriz_A.shape)
    reshaped_predictions = predictions.reshape(-1, future_steps, len(features))
    reshaped_predictions_A = predictions_A.reshape(-1, future_steps, len(features))
    print(reshaped_predictions,"--->",reshaped_predictions.shape)
    print(reshaped_predictions_A,"--->",reshaped_predictions_A.shape)
    
    arr_squeezed = np.squeeze(reshaped_predictions)
    arr_squeezed_A=np.squeeze(reshaped_predictions_A)
    predictions_unscaled = scaler.inverse_transform(arr_squeezed)
    predictions_unscaled_A=scaler.inverse_transform(arr_squeezed_A)#close es el [0]
    diff=predictions_unscaled_A[0][0]-data_array[-1:][0][0]
    
    predictions_unscaled_ajustado=predictions_unscaled-diff
    
    
    ##puro de aca para abajo
    data_index = data.index
    new_dates = pd.date_range(start=data_index[-1], periods=predictions_unscaled.shape[0] + 1, freq='1H')[1:]
    #new_dates=pd.date_range(start=data_index[-1], periods=2+1, freq='1H')[1:]
    new_index = pd.DatetimeIndex(list(data_index) + list(new_dates))
    print("Data Shape:", data.shape)
    print("Predictions Unscaled Shape:", predictions_unscaled.shape)
    print("Predictions Unscaled Ajustado Shape:", predictions_unscaled_ajustado.shape)
    concatenated_ant=np.concatenate([data, predictions_unscaled], axis=0)
    concatenated_adjusted=np.concatenate([data, predictions_unscaled_ajustado], axis=0)
    print("Data Index Length:", len(data_index))
    print("New Dates Length:", len(new_dates))
    print("New Index Length:", len(new_index))
    df_final_ant=pd.DataFrame(concatenated_ant,columns=features,index=new_index)
    df_final_ant['open'] = df_final_ant['close'].shift(1)
    df_final_ant=df_final_ant.dropna()
    print("DataFrame df_final_ant Shape:", df_final_ant.shape)
    print("DataFrame df_final_ant Head:\n", df_final_ant.head())
    renamed_data_final = df_final_ant.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
    
    last_100_rows_final = renamed_data_final.tail(50)
    mpf.plot(last_100_rows_final, type='candle', style='yahoo', warn_too_much_data=51, mav=(10,20), savefig='chart_1.png')
    
    df_final_ajustado=pd.DataFrame(concatenated_adjusted,columns=features,index=new_index)
    print("DataFrame df_final_ajustado Shape:", df_final_ajustado.shape)
    print("DataFrame df_final_ajustado Head:\n", df_final_ajustado.head())
    df_final_ajustado['open'] = df_final_ajustado['close'].shift(1)
    df_final_ajustado=df_final_ajustado.dropna()
    renamed_data_final_ajustado = df_final_ajustado.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
    last_100_rows_final_ajustado = renamed_data_final_ajustado.tail(50)
    mpf.plot(last_100_rows_final_ajustado , type='candle', style='yahoo', warn_too_much_data=51, mav=(10,20), savefig='chart_2.png')
    return predictions_unscaled_ajustado

def get_chart():
    # Obtener los datos de precios del EURUSD
    data = yf.download('EURUSD=X', period='1d', interval='1h')
    # Crear un gráfico de velas usando mplfinance
    mpf.plot(data, type='candle', volume=True, mav=(7, 21), savefig='chart.png')

# Función para enviar el gráfico a través de Telegram
def send_chart(entrenar):
    # Obtener el chat ID del usuario o grupo al que deseas enviar el gráfico
    chat_id = "1351057774"
    # Obtener el gráfico
    predictions= entrenamiento(entrenar)
    # Enviar el gráfico a través de Telegram
    with open('chart_1.png', 'rb') as f:
        bot.send_photo(chat_id, f)
        
    
    with open('chart_2.png','rb') as d:
        bot.send_photo(chat_id, d)
    # Esperar 4 horas antes de enviar el próximo gráfico
    return predictions
    

    

# Función para manejar el comando /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "¡Hola! Este es un bot para enviar gráficos del EURUSD cada hora. Para comenzar, usa el comando /entrenar si no descargaste el modelo entrenado o quieres volverlo a entrenar(es un poco demorado) de lo contrario escribe /predict_chart")

# Función para manejar el comando /send_chart
@bot.message_handler(commands=['predict_chart'])
def send_chart_message(message):
    predictions=send_chart(False)
    slopes=calculate_slopes(predictions)
    general_slope=calculate_general_slope(predictions)
    sell_signal = all(slope < 0 for _, _, slope in slopes) and general_slope < 0
    buy_signal = all(slope > 0 for _, _, slope in slopes) and general_slope > 0

    if sell_signal:
        bot.reply_to(message, "¡Vende! Todas las pendientes son negativas.")
    elif buy_signal:
        bot.reply_to(message, "¡Compra! Todas las pendientes son positivas.")
    else:
        bot.reply_to(message, "No hay una señal clara para operar en este momento.")
        
      
@bot.message_handler(commands=['predict_chart_hourly'])
def send_chart_message(message):
    diff_time=0
    while True:
        predictions=send_chart(False)
        slopes=calculate_slopes(predictions)
        general_slope=calculate_general_slope(predictions)
        sell_signal = all(slope < 0 for _, _, slope in slopes) and general_slope < 0
        buy_signal = all(slope > 0 for _, _, slope in slopes) and general_slope > 0
    
        if sell_signal:
            bot.reply_to(message, "¡Vende! Todas las pendientes son negativas.")
        elif buy_signal:
            bot.reply_to(message, "¡Compra! Todas las pendientes son positivas.")
        else:
            bot.reply_to(message, "No hay una señal clara para operar en este momento.")
        time.sleep(3600-diff_time)
      
@bot.message_handler(commands=['entrenar'])
def send_chart_message(message):
    predictions=send_chart(True) 
    slopes=calculate_slopes(predictions)
    general_slope=calculate_general_slope(predictions)
    sell_signal = all(slope < 0 for _, _, slope in slopes) and general_slope < 0
    buy_signal = all(slope > 0 for _, _, slope in slopes) and general_slope > 0

    if sell_signal:
        bot.reply_to(message, "¡Vende! Todas las pendientes son negativas.")
    elif buy_signal:
        bot.reply_to(message, "¡Compra! Todas las pendientes son positivas.")
    else:
        bot.reply_to(message, "No hay una señal clara para operar en este momento.")
   
    
    

# Ejecutar el bot
bot.polling()