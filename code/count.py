import csv

import pandas as pd

if __name__ == '__main__':
    # data = np.loadtxt(open("../data/train/train.csv"), delimiter=",", skiprows=1, usecols=(4))
    data = pd.read_csv(
        "../data/test/test.csv")  # , sep=",", dtype=[int, int, str, int, str, str, float, int, float, int, float, float, float, int]
    print(data.shape)
    data = data.values[0::, 0::]
    print(len(data))
    print(len(data[0]))
    for i in range(int(len(data))):
        if data[i][4] == 'France':
            data[i][4] = 0
        if data[i][4] == 'Spain':
            data[i][4] = 1
        if data[i][4] == 'Germany':
            data[i][4] = 2

        if data[i][5] == 'Male':
            data[i][5] = 0
        if data[i][5] == 'Female':
            data[i][5] = 1

    with open('../data/test/test_shaped.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
             "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"])
        for i in range(int(len(data))):
            writer.writerow(data[i])

# main()
