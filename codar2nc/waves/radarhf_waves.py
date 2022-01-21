from datetime import datetime
import json
from collections import OrderedDict
import psycopg2
import numpy as np
import pandas as pd


class Wave:
    """
    Class for reading and processing wls (waves) files from a Codar Combine.
    """

    def __init__(self, file_wls):
        """
        Constructor:

        :param: file_in, wls file to be read

        """
        self.range_max = 5  # TODO: look for this.

        content = [line.decode('utf-8').replace('%', '').replace('\n', '') for line in
                   open(file_wls, 'rb').readlines() if '%%' not in str(line)]

        metadata = [line for line in content if 'Table' not in line]
        metadata = dict([(line.split(':')[0], line.split(':')[1]) for line in metadata if ':' in str(line)])

        # Líneas inicial y final de las tablas:
        starts = np.arange(len(content))[['TableStart' in linea for linea in content]]
        ends = np.arange(len(content))[['TableEnd' in linea for linea in content]]
        lengths = ends - starts - 1

        # Linea que contiene el header:
        columns = np.arange(len(content))[['TableColumnTypes' in linea for linea in content]]

        tablas = []

        headers = [content[indice].split(':')[1].split() for indice in columns]
        for i in range(self.range_max):
            if lengths[i] != 0:
                start = starts[i] + 1
                end = ends[i]
                tablas.append(pd.DataFrame(np.array([linea.replace('"', '').split() for linea in content[start:end]]),
                                           columns=headers[i]))

        #  TIME MWHT MWPD WAVB WNDB ACNT DIST RCLL WDPT MTHD FLAG TYRS TMON TDAY THRS TMIN TSEC
        tipos = {'TIME': np.dtype(int),
                 'MWHT': np.dtype(float),
                 'MWPD': np.dtype(float),
                 'WAVB': np.dtype(float),
                 'WNDB': np.dtype(float),
                 'ACNT': np.dtype(int),
                 'DIST': np.dtype(float),
                 'RCLL': np.dtype(float),
                 'WDPT': np.dtype(int),
                 'MTHD': np.dtype(int),
                 'FLAG': np.dtype(int),
                 'TYRS': np.dtype(int),
                 'TMON': np.dtype(int),
                 'TDAY': np.dtype(int),
                 'THRS': np.dtype(int),
                 'TMIN': np.dtype(int),
                 'TSEC': np.dtype(int)
                 }

        tablas_with_type = [tabla.astype(tipos) for tabla in tablas]

        self.metadata = metadata
        self.headers = headers[0]
        self.tablas = tablas_with_type

    def get_last_row(self, rcell):

        if rcell >= self.range_max:
            print(f'Range Cell Index from 0 to {self.range_max - 1}')
            return None
        return self.tablas[rcell].tail(1)

    def get_last_time(self, rcell):
        last_row = self.get_last_row(rcell)
        return self.get_time(last_row)

    def get_time_by_index(self, rcell, row_index):
        if self.check_row_index_is_in_range(rcell, row_index):
            row = self.tablas[rcell].iloc[row_index - 1:row_index]
            return self.get_time(row)

    def get_time(self, row):
        year = row['TYRS'].values[0]
        month = row['TMON'].values[0]
        day = row['TDAY'].values[0]
        hour = row['THRS'].values[0]
        minute = row['TMIN'].values[0]
        second = row['TSEC'].values[0]
        return datetime(year, month, day, hour, minute, second)

    def get_wave_values_by_index(self, rcell, row_index):
        if self.check_row_index_is_in_range(rcell, row_index):
            row = self.tablas[rcell].iloc[row_index - 1:row_index]
            return self.get_wave_values(row)

    def get_last_wave_values(self, rcell):
        last_row = self.get_last_row(rcell)
        return self.get_waves_values(last_row)

    def get_wave_values(self, row):
        height = row['MWHT'].values[0]
        period = row['MWPD'].values[0]
        direction = row['WAVB'].values[0]
        return height, period, direction

    def check_row_index_is_in_range(self, rcell, row_index):
        max_rows = self.tablas[rcell].shape[0]
        if row_index > max_rows:
            print(f'Number max of rows = {max_rows}. Index must be less or equal. Your index = {row_index}')
            return False
        return True


def read_connection(input_file):
    try:
        with open(input_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        print(f'File not found: {input_file} ')
        if input('Do you want to create one (y/n)?') == 'n':
            quit()


def get_db_connection(db_json):
    database_data = read_connection(db_json)
    connection_string = 'host={0} port={1} dbname={2} user={3} password={4}'.format(database_data['host'],
                                                                                    database_data['port'],
                                                                                    database_data['dbname'],
                                                                                    database_data['user'],
                                                                                    database_data['password'])
    try:
        return psycopg2.connect(connection_string)
    except psycopg2.OperationalError as e:
        print('CAUTION: ERROR WHEN CONNECTING TO {0}'.format(database_data['host']))


def convert_into_dictionary(list_of_tuples):
    dictionary = {}
    for a, b in list_of_tuples:
        dictionary.setdefault(a, b)
    return dictionary


def wave2db(path, file_in):

    site_name = path.split('/')[-1]

    db_json_file = r'../pass/svr_dev_1.json'
    full_path_file = path + '/' + file_in
    wave = Wave(full_path_file)

    connection = get_db_connection(db_json_file)
    cursor = connection.cursor()
    sql = '''SELECT code, pk FROM  waves.sites ORDER BY pk ASC'''
    cursor.execute(sql)
    id_sites = convert_into_dictionary(cursor.fetchall())

    id_site = id_sites[site_name]
    for rcell in range(wave.range_max):
        for ntimes in range(1, wave.tablas[rcell].shape[0] + 1):
            print(f'Range = {rcell}: date: {wave.get_time_by_index(rcell, ntimes)} ---> '
                  f'{wave.get_wave_values_by_index(rcell, ntimes)}')
            date_sql = wave.get_time_by_index(rcell, ntimes).strftime('%Y-%m-%d %H:%M:00.00')
            print(date_sql)
            sql = '''SELECT * FROM  waves.values WHERE datetime = %s  AND fk_site = %s AND fk_range = %s'''
            params = (date_sql, id_site, rcell+1 )
            cursor.execute(sql, params)
            print(cursor.fetchall())
            existe = bool(cursor.rowcount)

            print('vamos----> ', existe)
            try:
                if not existe:
                    height, period, direction = wave.get_wave_values_by_index(rcell, ntimes)
                    sql = '''INSERT INTO waves.values(fk_site, fk_range, datetime, height, period, direction) 
                    VALUES(%s, %s, %s, %s, %s, %s)'''
                    params = (id_site, rcell+1, date_sql, height, period, direction)
                    cursor.execute(sql, params)
                    connection.commit()

                else:
                    print(f'VALUES: {params} already exists')
            except Exception as err:
                print(err)


    cursor.close()
    connection.close()


if __name__ == '__main__':
    file = r'WVLM_SILL_2021_11_01_0000.wls'
    path_in = r'../../datos/radarhf_tmp/wls/SILL'

    wave2db(path_in, file)
