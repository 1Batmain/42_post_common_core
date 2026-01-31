import csv
from utils import estimate_price 
import matplotlib.pyplot as plt


def parse_data():
    with open('data.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        mileages = []
        prices = []
        next(csv_reader)  # Skip header
        for row in csv_reader:
            mileages.append(float(row[0]))
            prices.append(float(row[1]))
    return mileages, prices


def train(mileages, prices, line, learning_rate=0.001, epochs=90):
    teta0 = 1
    teta1 = 1

    for epoch in range(epochs):
        tmpTeta0 = learning_rate * sum(estimate_price(mileages[i], teta0, teta1) - prices[i] for i in range(len(mileages))) / len(mileages)
        tmpTeta1 = learning_rate * sum((estimate_price(mileages[i], teta0, teta1) - prices[i]) * mileages[i] for i in range(len(mileages))) / len(mileages)
        teta0 = teta0 - tmpTeta0
        teta1 = teta1 - tmpTeta1
        if not epoch % 10:
            update_plot(mileages, prices, teta0, teta1, line, 0)
    update_plot(mileages, prices, teta0, teta1, line, 1)


    return teta0, teta1

def init_plot_results(mileages, prices, teta0 = 0, teta1 = 0):
    plt.ion()
    plt.scatter(mileages, prices, color='blue', label='Data Points')
    line_x = [min(mileages), max(mileages)]
    line_y = [estimate_price(x, teta0, teta1) for x in line_x]
    line, = plt.plot(line_x, line_y, color='red', label='Regression Line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression Result')
    plt.legend()
    return line

def update_plot(mileages, prices, teta0, teta1, line, last=0):
    line_x = [min(mileages), max(mileages)]
    line_y = [estimate_price(x, teta0, teta1) for x in line_x]
    line.set_ydata(line_y)
    plt.draw()
    plt.pause(1)
    if last == 1:
        plt.ioff()
        plt.show()


def main():
    print("Training started...")
    mileages, prices = parse_data()
    line = init_plot_results(mileages, prices)
    teta0, teta1 = train(mileages, prices, line)
    print(f"Training completed. teta0: {teta0}, teta1: {teta1}")

if __name__ == "__main__":
    main()

