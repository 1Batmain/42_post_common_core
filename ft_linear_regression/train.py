import csv
from utils import estimate_price 
import matplotlib.pyplot as plt


# def parse_data():
#     with open('data.csv', mode='r') as file:
#         csv_reader = csv.reader(file)
#         mileages = []
#         prices = []
#         next(csv_reader)  # Skip header
#         for row in csv_reader:
#             mileages.append(float(row[0]))
#             prices.append(float(row[1]))
#     return mileages, prices

def parse_data ():
    mileages = [0, 1 , 2, 3, 5 , 28]
    prices = [ 10, 12, 15, 17, 20, 30]
    return mileages, prices


def train(mileages, prices, learning_rate=0.001, epochs=500):
    teta0 = 1
    teta1 = 1

    for epoch in range(epochs):
        tmpTeta0 = learning_rate * sum(estimate_price(mileages[i], teta0, teta1) - prices[i] for i in range(len(mileages))) / len(mileages)
        tmpTeta1 = learning_rate * sum((estimate_price(mileages[i], teta0, teta1) - prices[i]) * mileages[i] for i in range(len(mileages))) / len(mileages)
        teta0 = teta0 - tmpTeta0
        teta1 = teta1 - tmpTeta1


    return teta0, teta1

def plot_results(mileages, prices, teta0 = 0, teta1 = 0):
    plt.scatter(mileages, prices, color='blue', label='Data Points')
    line_x = [min(mileages), max(mileages)]
    line_y = [estimate_price(x, teta0, teta1) for x in line_x]
    plt.plot(line_x, line_y, color='red', label='Regression Line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression Result')
    plt.legend()
    plt.show()



def main():
    print("Training started...")
    mileages, prices = parse_data()
    teta0, teta1 = train(mileages, prices)
    plot_results(mileages, prices, teta0, teta1)
    print(f"Training completed. teta0: {teta0}, teta1: {teta1}")

if __name__ == "__main__":
    main()

