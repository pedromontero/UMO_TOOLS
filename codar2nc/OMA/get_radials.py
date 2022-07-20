"""The object of this script is to copy radials from radial_tmp to RadialFiles
   to be used by radial2totalOMA.py for """
import sys
from datetime import datetime, timedelta
from os import path, getcwd
from shutil import copy


class Radials:
    def __init__(self):
        self.stations = ['PRIO', 'VILA', 'FIST', 'SILL', 'LPRO']
        self.working_folder = r'./datos/RadialFiles/.'
        self.origin_path = path.join('..', '..', 'datos', 'radarhf_tmp', 'ruv')
        self.prefix_file = 'RDLm'
        print(getcwd())

    def build_origin_file(self, station, datetime_origin):
        return datetime_origin.strftime(self.prefix_file + '_' + station + '_%Y_%m_%d_%H00.ruv')

    def get_full_origin_file(self, station, datetime_origin):
        return path.join(self.origin_path, station, self.build_origin_file(station, datetime_origin))

    def copy_radials(self, date):
        for station in self.stations:
            for date_origin_file in self.get_origin_files(date):
                full_origin_file = self.get_full_origin_file(station, date_origin_file)
                print(f'Copying {full_origin_file} to {working_folder}')
                try:
                    copy(full_origin_file, working_folder)
                    print('Success')
                except:
                    print('Unable', sys.exc_info()[0])

    @staticmethod
    def get_origin_files(date):
        return [date + timedelta(hours=hour) for hour in range(0, 24)]


if __name__ == '__main__':

    working_folder = './datos/RadialFiles'
    date_file = datetime(2022, 5, 1)
    radials = Radials()
    radials.copy_radials(date_file)

