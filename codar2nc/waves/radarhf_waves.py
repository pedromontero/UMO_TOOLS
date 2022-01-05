from datetime import datetime
import numpy as np
import pandas as pd


class Wave:
    """
    Class for reading and processing wls (waves) files from a Codar Combine. PMV
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
            print(f'Range Cell Index from 0 to {self.range_max-1}')
            return None
        return self.tablas[rcell].tail(1)

    def get_last_time(self, rcell):
        last_row = self.get_last_row(rcell)
        return self.get_time(last_row)

    def get_time_by_index(self, rcell, row_index):
        if self.check_row_index_is_in_range(rcell, row_index):
            row = self.tablas[rcell].iloc[row_index-1:row_index]
            return self.get_time(row)


    def get_time(self,  row):
        year = row['TYRS'].values[0]
        month = row['TMON'].values[0]
        day = row['TDAY'].values[0]
        hour = row['THRS'].values[0]
        minute = row['TMIN'].values[0]
        second = row['TSEC'].values[0]
        return datetime(year, month, day, hour, minute, second)

    def get_wave_values_by_index(self, rcell, row_index):
        if self.check_row_index_is_in_range(rcell, row_index):
            row = self.tablas[rcell].iloc[row_index-1:row_index]
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


def wave2db(path, file_in):
    full_path_file = path + '/' + file_in
    wave = Wave(full_path_file)
    for rcell in range(wave.range_max):
        for ntimes in range(1, wave.tablas[rcell].shape[0]+1):
            print(f'Range = {rcell}: date: {wave.get_time_by_index(rcell, ntimes)} ---> '
                  f'{wave.get_wave_values_by_index(rcell, ntimes)}')



if __name__ == '__main__':
    file = r'WVLM_VILA_2021_11_01_0000.wls'
    path_in = r'../../datos/radarhf_tmp/wls/VILA'

    wave2db(path_in, file)
