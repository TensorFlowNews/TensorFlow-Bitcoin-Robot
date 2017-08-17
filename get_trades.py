#coding:utf-8
import json
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_trades():
    re=requests.get('https://api.btctrade.com/api/trades?coin=btc')
    content=re.content
    trades=json.loads(content)
    date=[]
    price=[]
    amount=[]
    for trade in trades:
        date.append(trade['date'])
        price.append(trade['price'])
        amount.append(trade['amount'])
    return date,price,amount

def main():
    date,price,amount =get_trades()
    print(len(price))
    plt.figure()
    plt.plot(date, price)
    plt.show()

if __name__ == '__main__':
  main()


