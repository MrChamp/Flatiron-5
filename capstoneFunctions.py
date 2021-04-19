'''
Functions used in FlatIron data science capstone project

Sections:
-MySQL Connection
-Closing Price Column 
-Feeature Generation
-Exploratory Analysis
'''


'''MySQL Connection Functions'''
## Database connection functions
import mysql.connector
from mysql.connector import Error
# string, string, string, sting -> connection, string
# takes in a host, user, password, 
#    and database name and connects to db
def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print('MySQL Database connection successful')
    except Error as err:
        print(f"Error: '{err}'")
    
    return connection
# conn, sql cmd string -> mysql result, list
# returns a mysql query and column name list
def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cols = cursor.column_names
        return result, cols
    except Error as err:
        print(f"Error: '{err}'")

''' Add/Match Closing Data Functions'''    
    
## Stock data imports
import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import time

# !Note could refactor to generalize these two functions (1)!
# str, dateTime -> number
# returns the stocks closing price given an
#    options chain mine date
def mined_stock_close(ticker, minedate):
    start = minedate
    end = minedate + datetime.timedelta(days=1)
    stock = yf.Ticker(ticker)
    history = stock.history(start=minedate, end=end, interval = '1d')
    time.sleep(2)
    return history['Close']


# !Note could refactor to generalize these two functions (2)!
# # str, datetime -> number
# def mined_stock_close(ticker, minedate):
#     start = minedate
#     end = minedate + datetime.timedelta()
def add_close_series(two_col_df):
    # This is not the most efficient method
    # there is a lot of 'double counting' or extra work
    # being done
    
    # define current ticker and mine date
    this_ticker = two_col_df[0]
    this_mine_date = two_col_df[1]
    # get closing date from ticker and mine date
    this_close = mined_stock_close(this_ticker, this_mine_date)
    # create dataframe with ticker id and close value
    ticker_close = pd.DataFrame()
    ticker_close['ticker'] = this_ticker
    ticker_close['close'] = this_close
    
    return ticker_close

# string, df, str -> number
# matches and returns the closing price of a stock given ticker
#    to be applied to a dataframe
def match_close(ticker, df, col):
    # this is inefficient because it takes in the df
    # each time... should set df as global or something
    return df[col].loc[df.index == ticker][0]


'''
Feature Generation
'''
from scipy.stats import norm

# df[number, number, number, number, number] -> number
# theoretical cost derived from Black-Scholes formula:
def bs_cost(df):
    # S = closing price
    # X = strike price
    # sigma = implied volatility
    # r = us treasury bill rate = .0002
    # q = dividend yield = 0
    # ttm = time to maturity in days
    # type = option type
    S = df[0]
    X = df[1]
    sigma = df[2]
    r = .0002
    q = 0
    ttm = df[3]
    call = df[4]
    
    
    b = r-q
    t = ttm/365.25
    
    d1 = (np.log(S/X) + (b + (sigma**2)/2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    if call:
        price = S*np.exp((b-r)*t) * norm.cdf(d1) - X * np.exp(-r*t) * norm.cdf(d2)
    else:
        price = (X * np.exp(-r*t) * norm.cdf(-d2) - S * np.exp((b-r)*t) * norm.cdf(-d1))
    return price



# df[number, number, bool] -> number 
# given the strke price, close price and call/put bool
#    will return the "moneyness" of option
def moneyness(df):
    stock = df[0]
    strike = df[1]
    is_call = df[2]
    if is_call:
        return stock/strike
    else:
        return strike/stock

'''
Exploratory Analysis
'''
import matplotlib.pyplot as plt
# pandas dataframe, string -> histogram
# returns a histogram of a pandas series restricted to the data
#    within one standard deviation of the mean
def data_one_std(df, col_name):
    df_col = df[col_name]
    pared = df_col.loc[(df_col < df_col.mean()+df_col.std()) & (df_col > df_col.mean()-df_col.std())]
    return pared

'''
GAIN
'''
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# -----------------------------------------------------------------------------
# These functions are from the utilz.py library

#
#
def normalization(data, parameters=None):
    # Parameters
    _, dim = data.shape
    norm_data = data.copy()
    if parameters is None:
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                       'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
    # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
    norm_parameters = parameters    
    return norm_data, norm_parameters


#
#
def renormalization(norm_data, norm_parameters):
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    return renorm_data

#
#
def rounding(imputed_data, data_x):
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()    
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data

#
#
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)

#
#
def binary_sampler(p, rows, cols):
    unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix

#
#
def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size = [rows, cols]) 

#
#
def sample_batch_index(total, batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx
#-----------------------------------------------------------------------------

#
#
def gain(data, gain_params):
    # generate missing values in data_x
    
    # mask matrix:
    mask = 1-np.isnan(data)
    
    # system parameters:
    batch_size = gain_params['batch_size']
    hint_rate = gain_params['hint_rate']
    alpha = gain_params['alpha']
    iterations = gain_params['iterations']
    
    # other parameters:
    no, dim = data.shape
    
    # hidden stat dims
    h_dim = int(dim)
    
    # normalization
    normed_d, normed_p = normalization(data)
    norm_data = np.nan_to_num(normed_d,0)
    
    # data vector
    X = tf.placeholder(tf.float32, shape = [None, dim])
    # mask vector
    M = tf.placeholder(tf.float32, shape = [None, dim]) 
    # hint vector
    H = tf.placeholder(tf.float32, shape = [None, dim])
    
    # discriminator variables
    # tensorflow variable to be changed/updated as GAN "competes"
    
    # data + hint
    d_w1 = tf.Variable(xavier_init([dim*2, h_dim]))
    d_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    d_w2 = tf.Variable(xavier_init([h_dim, h_dim]))
    d_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    d_w3 = tf.Variable(xavier_init([h_dim, dim]))
    d_b3 = tf.Variable(tf.zeros(shape=[dim]))
    
    d_theta = [d_w1, d_w2, d_w3, d_b1, d_b2, d_b3]
    
    # generator variables
    g_w1 = tf.Variable(xavier_init([dim*2, h_dim]))
    g_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    g_w2 = tf.Variable(xavier_init([h_dim, h_dim]))
    g_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    g_w3 = tf.Variable(xavier_init([h_dim, dim]))
    g_b3 = tf.Variable(tf.zeros(shape = [dim]))
    
    g_theta = [g_w1, g_w2, g_w3, g_b1, g_b2, g_b3]
    
    # generator function
    #
    #
    def generator(x, m):
        # cncat mask with data
        inputs = tf.concat(values = [x, m], axis=1)
        # neural network
        g_h1 = tf.nn.relu(tf.matmul(inputs, g_w1) + g_b1)
        g_h2 = tf.nn.relu(tf.matmul(g_h1, g_w2) + g_b2)
        
        g_prob = tf.nn.sigmoid(tf.matmul(g_h2, g_w3) + g_b3)
        return g_prob
    
    # discriminator function
    #
    #
    def discriminator(x, h):
        inputs = tf.concat(values=[x,h], axis=1)
        # neural network
        d_h1 = tf.nn.relu(tf.matmul(inputs, d_w1) + d_b1)
        d_h2 = tf.nn.relu(tf.matmul(d_h1, d_w2) + d_b2)
        d_logit = tf.matmul(d_h2, d_w3) + d_b3
        d_prob = tf.nn.sigmoid(d_logit)
        return d_prob
    
    # run generator and discriminator
    g_prob = generator(X,M)
    # combine with observed data
    x_hat = X*M + g_prob * (1-M)
    d_prob = discriminator(x_hat, H)
    
    # GAIN loss:
    d_loss_temp = -tf.reduce_mean(M * tf.log(d_prob + 1e-8)\
                                 + (1-M) * tf.log(1 - d_prob + 1e-8))
    
    g_loss_temp = -tf.reduce_mean((1-M) * tf.log(d_prob + 1e-8))
    
    mse_loss = tf.reduce_mean((M*X-M*g_prob)**2)/tf.log(d_prob + 1e-8)

    d_loss = d_loss_temp
    g_loss = g_loss_temp + alpha + mse_loss
    
    # GAIN solver:
    d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_theta)
    g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_theta)
    
    # iterate
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for it in tqdm(range(iterations)):
        # sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data[batch_idx, :]
        M_mb = mask[batch_idx, :]
        
        # sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # sample hint vectors
        H_mb_tmp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb + H_mb_tmp
        
        X_mb = M_mb + X_mb + (1-M_mb) * Z_mb
        
        _, d_loss_cur = sess.run([d_solver, d_loss_temp], feed_dict = {M: M_mb, X:X_mb, H:H_mb})
        _, g_loss_cur, mse_loss_curr = sess.run([g_solver, g_loss_temp, mse_loss], feed_dict = {X:X_mb, M:M_mb, H:H_mb})
    
    # imputed data
    Z_mb = uniform_sampler(0,0.01, no, dim)
    M_mb = mask
    X_mb = norm_data
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    
    imputed_data = sess.run([g_prob], feed_dict = {X:X_mb, M:M_mb})[0]
    
    imputed_data = mask * norm_data + (1-mask) * imputed_data
    
    # renormalize
    imputed_data = renormalization(imputed_data, normed_p)
    
    
    # rounding
    imputed_data = rounding(imputed_data, data)
    
    return imputed_data