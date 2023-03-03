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


if __name__ == '__main__':
    dtsn_data = load_DTSN_data()
    # unique_dtsn = pydash.map_(dtsn_data, lambda x: x[22:26])
    # unique_dtsn2 = pydash.uniq(unique_dtsn)
    # DTSN_FLARES = dict.fromkeys(unique_dtsn2, 0)


    for row in dtsn_data:
        cleansed_row = row.split(' ')
        cleansed_row = pydash.filter_(cleansed_row, lambda x: len(x) != 0)
        DAT_DTSN = int(cleansed_row[4])
        DAT_DATE = datetime.strptime(f'{cleansed_row[0]}-{cleansed_row[1]}-{cleansed_row[2]}', '%Y-%m-%d')
        delta_t = DAT_DATE + timedelta(days=1)
        number_of_flares_that_day = df[(df['t-inicio'] >= str(DAT_DATE)) & (df['t-inicio'] <= str(delta_t))].shape[0]
        daily_flare_count.append(number_of_flares_that_day)
        daily_dat_dtsn.append(DAT_DTSN)
        unique_dtsn.add(DAT_DTSN)

unique_dtsn = list(unique_dtsn)
dtsn_flares = dict.fromkeys(unique_dtsn, 0)
for (dtsn, flare_count) in zip(daily_dat_dtsn, daily_flare_count):
    dtsn_flares[dtsn] += flare_count

pprint(dtsn_flares)


width = 1.0     # gives histogram aspect to the bar diagram
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 7)
plt.bar(dtsn_flares.keys(), np.array(list(dtsn_flares.values())), width, color='g')
plt.ylabel('Flares')
plt.xlabel('Sunspots')
plt.title('Flare distribution by sunspot number')
plt.grid(linestyle='--', linewidth=.7)
fig.savefig('test2png.png', dpi=100)
plt.show()
