import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tkinter as tk
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
stock_list = 'INDF.JK'
try:
    datasets = yf.download(stock_list, start, end)
    datasets.to_csv('datasets.CSV')
except:
    print("Failed to download datasets. Assuming using last datasets.")
try:
    df = pd.read_csv('datasets.CSV')
except:
    raise exception("Datasets is missing, please connect to the internet to get the datasets")
df.drop('Volume',axis=1,inplace=True)
df.drop('Adj Close',axis=1,inplace=True)
df.drop('Date',axis=1,inplace=True)
x = df.iloc[:,0:3]
y = df.iloc[:,3]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=random.randrange(1,10000))
model = LinearRegression()
model.fit(x_train,y_train)
####################################################
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()
label_Title = tk.Label(root, text='Stock Prediction using SKLearn')
label_Title.config(font=('Arial', 20))
canvas1.create_window(250, 50, window=label_Title)
label1 = tk.Label(root, text='Type Open Value: ')
canvas1.create_window(140, 100, window=label1)
entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)
label2 = tk.Label(root, text='Type High Value: ')
canvas1.create_window(140, 120, window=label2)
entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)
label3 = tk.Label(root, text='Type Low Value: ')
canvas1.create_window(140, 140, window=label3)
entry3 = tk.Entry (root) # create 3rd entry box
canvas1.create_window(270, 140, window=entry3)
def Close_value(): 
    global New_Open #our 1st input variable
    New_Open = float(entry1.get()) 
    
    global New_High #our 2nd input variable
    New_High = float(entry2.get()) 
    
    global New_Low
    New_Low = float(entry3.get())
    
    Prediction_result  = ('Predicted Close Value : ', model.predict([[New_Open ,New_High, New_Low]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
button1 = tk.Button (root, text='Predict Close Value',command=Close_value, bg='orange') # button to call the 'values' command above
canvas1.create_window(270, 170, window=button1)
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['Open'].astype(float),df['Close'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['Close Value']) 
ax3.set_xlabel('Open')
ax3.set_title('Open Vs. Close Value')
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['High'].astype(float),df['Close'].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax4.legend(['Close Value']) 
ax4.set_xlabel('High')
ax4.set_title('High Vs. Close Value')
figure5 = plt.Figure(figsize=(5,4), dpi=100)
ax5 = figure5.add_subplot(111)
ax5.scatter(df['Low'].astype(float),df['Close'].astype(float), color = 'b')
scatter5 = FigureCanvasTkAgg(figure5, root) 
scatter5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax5.legend(['Close Value']) 
ax5.set_xlabel('Low')
ax5.set_title('Low Vs. Close Value')
root.mainloop()