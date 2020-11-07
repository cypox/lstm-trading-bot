import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class Dataset:
  def __init__(self, source='yahoo', start='2000-01-01', end=None):
    if end is None:
      end = datetime.today().strftime('%Y-%m-%d')
    self.start = start
    self.end = end
    self.source = source
    self.x_train = []
    self.y_train = []
    self.tickers = set()
    self.df = pd.DataFrame()
    
    #parameters
    self.split = 0.8
  
  def add_ticker(self, ticker):
    self.tickers.add(ticker)
    ticker_df = web.DataReader(ticker, data_source=self.source, start=self.start, end=self.end)
    ticker_df.reset_index(inplace=True)
    ticker_df.set_index('Date', inplace=True)
    #TODO: include these later
    #ticker_df['{}_HL_pct_diff'.format(ticker)] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Low']
    #ticker_df['{}_daily_pct_chng'.format(ticker)] = (ticker_df['Close'] - ticker_df['Open']) / ticker_df['Open']
    ticker_df.rename(columns={'Adj Close':ticker}, inplace=True)
    ticker_df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
    ticker_df = ticker_df.replace([np.inf, -np.inf, np.nan], 0)
    if self.df.empty:
      self.df = ticker_df
    else:
      self.df = self.df.join(ticker_df, how='outer')
  
  def get_last_state(self, memory, transform=None):
    state = self.df[-memory:].values
    if transform is not None:
      state = transform.fit_transform(state)
    state = np.array(state)
    return state
  
  def generate_dataset(self, memory, transform=None):
    dataset = self.df.values
    training_data_len = math.ceil(len(dataset) * self.split)

    if transform is not None:
      dataset = transform.fit_transform(dataset)
    
    x_patches = []
    y_patches = []

    for i in range(memory, len(dataset)):
      x_patches.append(dataset[i-memory:i, :])
      y_patches.append(dataset[i, :])
    
    x_patches, y_patches = np.array(x_patches), np.array(y_patches)
    x_patches, y_patches = shuffle(x_patches, y_patches, random_state=0)

    x_train = x_patches[0:training_data_len, :]
    y_train = y_patches[0:training_data_len, :]
    x_test = x_patches[training_data_len:, :]
    y_test = y_patches[training_data_len:, :]
    
    """
    train_data = dataset[0:training_data_len, :]
    test_data = dataset[training_data_len:, :]

    #split the data in x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(memory, len(train_data)):
      x_train.append(train_data[i-memory:i, :])
      y_train.append(train_data[i, :])
    x_train, y_train = np.array(x_train), np.array(y_train)
    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #create the dataset x_test, y_test
    x_test = []
    y_test = []

    for i in range(memory, len(test_data)):
      x_test.append(test_data[i-memory:i, :])
      y_test.append(test_data[i, :])

    #convert the data into a numpy array
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    #reshape the data for the LSTM layer (add 3rd dimension)
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    """

    return x_train, y_train, x_test, y_test

  def preview(self):
    print(self.df.head())
    print(self.df.tail())

class Trader:
  def __init__(self, predictor_fct, transformer = None):
    self.balance = 10000
    self.predict = predictor_fct
    self.transformer = transformer
    return

  def process(self, state):
    #TODO: multiple tickers in state
    current_price = state[-1]
    state = np.expand_dims(state, 0)
    predicted_price = self.predict(state)
    if self.transformer is not None:
      predicted_price = self.transformer.inverse_transform(predicted_price)
      current_price = self.transformer.inverse_transform(state[0])[-1]
    percent_change = 100 * (predicted_price - current_price) / current_price
    print("Current price is {}$. Predicted price is {}$. Difference {:.2f}%.".format(current_price.item(), predicted_price.item(), percent_change.item()))
    if percent_change > 2:
      print("You should BUY.")
    elif percent_change > -1:
      print("You should HOLD.")
    else:
      print("You should SELL.")


def build_model(memory, features):
  #build the LSTM model
  model = Sequential()
  model.add(LSTM(64, return_sequences=True, input_shape=(memory, features)))
  model.add(LSTM(64, return_sequences=True))
  model.add(LSTM(32, return_sequences=False))
  model.add(Dense(32))
  model.add(Dense(16))
  model.add(Dense(features))
  model.summary()
  #compile the model
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model


generator = Dataset(end='2020-08-31')
generator.add_ticker('TSLA')
#generator.add_ticker('MSFT')
#generator.add_ticker('GOOGL')
#generator.add_ticker('AMZN')
#generator.add_ticker('AAPL')
generator.preview()

memory = 60
features = len(generator.tickers) * 1 # TODO: multiply by number of featuer per ticker

scaler = MinMaxScaler(feature_range=(0, 1))
x_train, y_train, x_test, y_test = generator.generate_dataset(memory, scaler)

model = build_model(memory, features)

model.fit(x_train, y_train, batch_size=1, epochs=1)

#get the models predicted price values
predictions = model.predict(x_test)

#get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(f'RMSE: {rmse}')

truth = scaler.inverse_transform(y_test)
predicted = scaler.inverse_transform(predictions)

for i in range(features):
  #visualize the data
  plt.figure(figsize=(16, 8))
  plt.title('Model')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price USD ($)', fontsize=18)
  #plt.plot(y_train[:,0])
  plt.plot(truth[:,i])
  plt.plot(predicted[:,i])
  plt.legend(['Val', 'Predictions'], loc='lower right')
  plt.show()

generator = Dataset(end='2020-11-05')
generator.add_ticker('TSLA')
t = Trader(model.predict, scaler)
state = generator.get_last_state(memory, scaler)
t.process(state)
