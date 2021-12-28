import os
import re
import shutil
from datetime import datetime, timedelta
from create_dir_structure import FolderTree
from tuv2nc import tuv2nc
import getradarfiles


def check_nc_file(full_file):
    if os.path.isfile(full_file):
        print(f'existe {full_file} ')
        return True
    else:
        print(f'No existe {full_file}')
        return False


class Tuv2Nc:
    def __init__(self, data_folder, system):
        self.system = system
        self.tuv_folder = os.path.join(data_folder, 'radarhf_tmp', 'tuv')
        self.root_folder = os.path.join(data_folder, 'radarhf/dev/RadarOnRAIA/Totals/v2.2')
        self.nc_folder = os.path.join(data_folder, 'radarhf_tmp', 'nc', 'total')

    def check_nc_files(self, thredds):

        for file in os.listdir(self.tuv_folder):
            site = re.findall("[A-Z]{4}", file.split('/')[-1])[0]
            day = datetime.strptime('%s%s%s%s' % tuple(re.findall("\d+", file.split('/')[-1])), '%Y%m%d%H%M')

            thredds.make_total_folder( day)
            full_file = os.path.join(self.root_folder, thredds.get_full_total_file_nc( day))
            if not check_nc_file(full_file):
                print(f'---------------- vou crear {full_file}')

                day_hour_before = day + timedelta(hours=-1)
                file_previous_hour = os.path.join(self.root_folder, thredds.get_full_total_file_nc( day_hour_before))
                if check_nc_file(file_previous_hour):
                    shutil.copy(file_previous_hour, self.nc_folder)
                day_2hour_previous = day + timedelta(hours=-2)
                file_2hour_previous = os.path.join(self.root_folder, thredds.get_full_total_file_nc( day_2hour_previous))
                if check_nc_file(file_2hour_previous):
                    shutil.copy(file_2hour_previous, self.nc_folder)
                print(f'---vou necesitar {file_previous_hour} e {file_2hour_previous}')
                try:
                    print(self.tuv_folder, self.nc_folder, file)
                    tuv2nc(self.tuv_folder, self.nc_folder, file)
                    print(os.path.join(self.nc_folder, self.get_name('Total', day)), full_file)

                except KeyError as e:
                    print("Error: KeyError", e)
                except:
                    print("Another exception")
                else:
                    shutil.move(os.path.join(self.nc_folder, self.get_name('Total', day)), full_file)
                shutil.rmtree(self.nc_folder)
                os.makedirs(self.nc_folder)


def tuv2thredds(data_folder):
    system = 'Galicia'

    tuv2nc = Tuv2Nc(data_folder, system)
    thredds = FolderTree(tuv2nc.root_folder)
    tuv2nc.check_nc_files(thredds)


if __name__ == '__main__':
    data_folder = r'../../datos/'
    # getradarfiles.get_radar_files(data_folder)

    tuv2thredds(data_folder)
