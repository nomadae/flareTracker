"""

Download Daily Total Sunspot Number (DTSN)

"""
import os
import urllib.request
from datetime import datetime

DTSN_FILENAME = "SN_d_tot_V2.0.txt"
LAST_DATE_STORED_FILENAME = "last_downloaded.dat"


def find_last_date(data: list) -> datetime:
    i = -1
    while 1:
        if len(data[i]) != 0:
            l = data[i].split(' ')
            date = l[:3]
            s = "-".join(date)
            return datetime.strptime(s, '%Y-%m-%d')
        else:
            i -= 1


def get_last_stored_date() -> datetime | None:
    l = [x for x in os.listdir('.') if x == LAST_DATE_STORED_FILENAME]
    try:
        s = l[0]
    except IndexError:
        s = ''
    if len(s) > 0:
        with open(LAST_DATE_STORED_FILENAME, 'r') as f:
            d = f.readlines()
            date_str = d[0]
            date_str = date_str.strip()
            return datetime.strptime(date_str, '%Y-%m-%d')
    else:
        return None


def main():
    url = "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"
    with urllib.request.urlopen(url) as f:
        data = f.read().decode('utf-8').split('\n')

    latest_update = find_last_date(data)

    if (get_last_stored_date()):
        last_downloaded = get_last_stored_date()
        if latest_update > last_downloaded:
            print(latest_update)
            print(last_downloaded)
            print('Hay que actualizar el archivo')
        else:
            print(latest_update)
            print(last_downloaded)
            print(f'No hay informacion nueva desde: {latest_update}')

    else:
        print(f'No se encontró información descargada, se prodecerá a almacenar en {DTSN_FILENAME}')
        with open(DTSN_FILENAME, 'w') as file:
            for row in data:
                file.write(row + '\n')
        with open(LAST_DATE_STORED_FILENAME, 'w') as file:
            s = datetime.today().strftime('%Y-%m-%d')
            file.write(s + '\n')


if __name__ == '__main__':
    main()
