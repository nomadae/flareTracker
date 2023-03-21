from datetime import datetime, timedelta
from pprint import pprint

import numpy as np
import pandas as pd
import pydash
from matplotlib import pyplot as plt
import matplotlib

DTSN_FILENAME = "SN_d_tot_V2.0.txt"

DTSN_BEGIN = 11
DTSN_END = 72
DATE_BEGIN = 1976
YEARS_TO_SEARCH = [x for x in range(DATE_BEGIN, 2022, 1)]

daily_dat_dtsn = []
daily_flare_count = []
unique_dtsn = set()

df = pd.read_csv('./data/flares-goes-x-ray-unified.dat', sep='\t')
print(df.columns)
print()

df['t-inicio'] = pd.to_datetime(df['t-inicio'])
df['t-max'] = pd.to_datetime(df['t-max'])
df['t-fin'] = pd.to_datetime(df['t-fin'])
df = df.drop(['unknown'], axis=1)


def load_DTSN_data() -> list:
    recent_data = []
    with open(DTSN_FILENAME, 'r') as f:
        data = f.readlines()
        for year in YEARS_TO_SEARCH:
            recent_data.append(pydash.filter_(data, lambda x: x[:4] == str(year)))  #  x[:4] corresponde al año de la observacion ej. 2014
        recent_data = pydash.flatten_deep(recent_data)
        recent_data = pydash.map_(recent_data, lambda x: x.strip())

    return recent_data

def load_DTSN_by_category() -> list:
    categories = ["A", "B", "C", "M", "X"]




if __name__ == '__main__':
    dtsn_data = load_DTSN_data()
    # unique_dtsn = pydash.map_(dtsn_data, lambda x: x[22:26])
    # unique_dtsn2 = pydash.uniq(unique_dtsn)
    # DTSN_FLARES = dict.fromkeys(unique_dtsn2, 0)

    categories = ["A", "B", "C", "M", "X"]
    A, B, C, M, X = [], [], [], [], []
    A_dtsn, B_dtsn, C_dtsn, M_dtsn, X_dtsn = [], [], [], [], []
    A_dtsn_set, B_dtsn_set, C_dtsn_set, M_dtsn_set, X_dtsn_set = set(), set(), set(), set(), set()


    for row in dtsn_data:
        cleansed_row = row.split(' ')
        cleansed_row = pydash.filter_(cleansed_row, lambda x: len(x) != 0)
        DAT_DTSN = int(cleansed_row[4])
        DAT_DATE = datetime.strptime(f'{cleansed_row[0]}-{cleansed_row[1]}-{cleansed_row[2]}', '%Y-%m-%d')
        delta_t = DAT_DATE + timedelta(days=1)
        number_of_flares_that_day = df[(df['t-inicio'] >= str(DAT_DATE)) & (df['t-inicio'] <= str(delta_t))].shape[0]
        if number_of_flares_that_day > 0:
            flares_that_day = df[(df['t-inicio'] >= str(DAT_DATE)) & (df['t-inicio'] <= str(delta_t))]
            for cat in categories:
                category_flare_number = flares_that_day[flares_that_day["Clase"] == cat].shape[0]
                if category_flare_number > 0:
                    if cat == 'A':
                        A.append(category_flare_number)
                        A_dtsn.append(DAT_DTSN)
                        A_dtsn_set.add(DAT_DTSN)
                    if cat == 'B':
                        B.append(category_flare_number)
                        B_dtsn.append(DAT_DTSN)
                        B_dtsn_set.add(DAT_DTSN)
                    if cat == 'C':
                        C.append(category_flare_number)
                        C_dtsn.append(DAT_DTSN)
                        C_dtsn_set.add(DAT_DTSN)
                    if cat == 'M':
                        M.append(category_flare_number)
                        M_dtsn.append(DAT_DTSN)
                        M_dtsn_set.add(DAT_DTSN)
                    if cat == 'X':
                        X.append(category_flare_number)
                        X_dtsn.append(DAT_DTSN)
                        X_dtsn_set.add(DAT_DTSN)
        # daily_flare_count.append(number_of_flares_that_day)
        # daily_dat_dtsn.append(DAT_DTSN)
        # unique_dtsn.add(DAT_DTSN)

# for cat in categories:

    A_dtsn_set = list(A_dtsn_set)
    dtsn_flares_A = dict.fromkeys(A_dtsn_set, 0)
    for (dtsn, flare_count) in zip(A_dtsn, A):
        dtsn_flares_A[dtsn] += flare_count

    B_dtsn_set = list(B_dtsn_set)
    dtsn_flares_B = dict.fromkeys(B_dtsn_set, 0)
    for (dtsn, flare_count) in zip(B_dtsn, B):
        dtsn_flares_B[dtsn] += flare_count

    C_dtsn_set = list(C_dtsn_set)
    dtsn_flares_C = dict.fromkeys(C_dtsn_set, 0)
    for (dtsn, flare_count) in zip(C_dtsn, C):
        dtsn_flares_C[dtsn] += flare_count

    M_dtsn_set = list(M_dtsn_set)
    dtsn_flares_M = dict.fromkeys(M_dtsn_set, 0)
    for (dtsn, flare_count) in zip(M_dtsn, M):
        dtsn_flares_M[dtsn] += flare_count

    X_dtsn_set = list(X_dtsn_set)
    dtsn_flares_X = dict.fromkeys(X_dtsn_set, 0)
    for (dtsn, flare_count) in zip(X_dtsn, X):
        dtsn_flares_X[dtsn] += flare_count

    pprint(dtsn_flares_A)
    pprint(dtsn_flares_B)
    pprint(dtsn_flares_C)
    pprint(dtsn_flares_M)
    pprint(dtsn_flares_X)

    def plot_A_histogram():
        width = 1.0     # gives histogram aspect to the bar diagram
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 7)
        plt.bar(dtsn_flares_A.keys(), np.array(list(dtsn_flares_A.values())), width, color='g')
        plt.ylabel('Flares')
        plt.xlabel('Sunspots')
        plt.title('Flare distribution by sunspot number\nCategory A')
        plt.grid(linestyle='--', linewidth=.7)
        fig.savefig('test2png_A.png', dpi=100)
        plt.show()

    def plot_B_histogram():
        width = 1.0     # gives histogram aspect to the bar diagram
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 7)
        plt.bar(dtsn_flares_B.keys(), np.array(list(dtsn_flares_B.values())), width, color='g')
        plt.ylabel('Flares')
        plt.xlabel('Sunspots')
        plt.title('Flare distribution by sunspot number\nCategory B')
        plt.grid(linestyle='--', linewidth=.7)
        fig.savefig('test2png_B.png', dpi=100)
        plt.show()

    def plot_C_histogram():
        width = 1.0     # gives histogram aspect to the bar diagram
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 7)
        plt.bar(dtsn_flares_C.keys(), np.array(list(dtsn_flares_C.values())), width, color='g')
        plt.ylabel('Flares')
        plt.xlabel('Sunspots')
        plt.title('Flare distribution by sunspot number\nCategory C')
        plt.grid(linestyle='--', linewidth=.7)
        fig.savefig('test2png_C.png', dpi=100)
        plt.show()

    def plot_M_histogram():
        width = 1.0     # gives histogram aspect to the bar diagram
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 7)
        plt.bar(dtsn_flares_M.keys(), np.array(list(dtsn_flares_M.values())), width, color='g')
        plt.ylabel('Flares')
        plt.xlabel('Sunspots')
        plt.title('Flare distribution by sunspot number\nCategory M')
        plt.grid(linestyle='--', linewidth=.7)
        fig.savefig('test2png_M.png', dpi=100)
        plt.show()
    def plot_X_histogram():
        width = 1.0     # gives histogram aspect to the bar diagram
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 7)
        plt.bar(dtsn_flares_X.keys(), np.array(list(dtsn_flares_X.values())), width, color='g')
        plt.ylabel('Flares')
        plt.xlabel('Sunspots')
        plt.title('Flare distribution by sunspot number\nCategory X')
        plt.grid(linestyle='--', linewidth=.7)
        fig.savefig('test2png_X.png', dpi=100)
        plt.show()

    #plot_A_histogram()
    #plot_B_histogram()
    #plot_C_histogram()
    #plot_M_histogram()
    #plot_X_histogram()

    def plot_stacked_bar():
        dtsn_flares = {  # keys are the DTSNs and values are the flare count
            "A": dtsn_flares_A,
            "B": dtsn_flares_B,
            "C": dtsn_flares_C,
            "M": dtsn_flares_M,
            "X": dtsn_flares_X,
        }
        max_ = -999999999.
        for class_, category_d in dtsn_flares.items():
            if len(category_d.keys()) > max_:
                max_cat = class_

        x_ax = list(dtsn_flares[max_cat].keys())
        ys = {"A": [], "B": [], "C": [], "M": [],"X": []}
        for c in categories:
            d = dtsn_flares[c]
            for x in x_ax:
                try:
                    ys[c].append(d[x])
                except KeyError:
                    ys[c].append(0)
        print()

        m_flare_b = list(np.add(ys['B'], ys['C']))
        x_flare_b = list(np.add(m_flare_b, ys['M']))

        width = 1.0     # gives histogram aspect to the bar diagram
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 7)

        plt.bar(x_ax, ys['B'], width, color='b')
        plt.bar(x_ax, ys['C'], width, bottom=ys['B'], color='g')
        plt.bar(x_ax, ys['M'], width, bottom=m_flare_b, color='y')
        plt.bar(x_ax, ys['X'], width, bottom=x_flare_b, color='r')
        plt.xlabel('Sunspot Number')
        plt.ylabel('Flares')
        plt.legend(["B", "C", "M", "X"])
        plt.grid(linestyle='--', linewidth=.7)
        plt.title("Flare occurrence by sunspot number\nand flare type")
        fig.savefig('dtsn_flare_by_category.png', dpi=100)
        plt.show()

    plot_stacked_bar()
