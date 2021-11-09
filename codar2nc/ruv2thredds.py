import os
import re
import shutil
from datetime import datetime, timedelta
from create_dir_structure import FolderTree, get_radial_name
from ruv2nc import ruv2nc


def check_nc_file(full_file):
    if os.path.isfile(full_file):
        print(f'existe {full_file} ')
        return True
    else:
        print(f'No existe {full_file}')
        return False


class Ruv2Nc:
    def __init__(self, data_folder):
        self.ruv_folder = os.path.join(data_folder, 'radials')
        self.root_folder = os.path.join(data_folder, 'thredds')
        self.nc_folder = os.path.join(data_folder, 'nc')

    def check_nc_files(self, thredds):

        for file in os.listdir(self.ruv_folder):
            site = re.findall("[A-Z]{4}", file.split('/')[-1])[0]
            day = datetime.strptime('%s%s%s%s' % tuple(re.findall("\d+", file.split('/')[-1])), '%Y%m%d%H%M')

            thredds.make_radial_folder(site, day)
            full_file = os.path.join(self.root_folder, thredds.get_full_file_nc(site, day))
            if not check_nc_file(full_file):
                print(f'---------------- voy a crear {full_file}')

                day_hour_before = day + timedelta(hours=-1)
                file_previous_hour = os.path.join(self.root_folder,thredds.get_full_file_nc(site, day_hour_before))
                if check_nc_file(file_previous_hour):
                    shutil.copy(file_previous_hour, self.nc_folder)
                day_2hour_previous = day + timedelta(hours=-2)
                file_2hour_previous = os.path.join(self.root_folder,thredds.get_full_file_nc(site, day_2hour_previous))
                if check_nc_file(file_2hour_previous):
                    shutil.copy(file_2hour_previous, self.nc_folder)
                print(f'---vou necesitar {file_previous_hour} e {file_2hour_previous}')
                ruv2nc(self.ruv_folder, self.nc_folder, file)
                print(os.path.join(self.nc_folder,get_radial_name(site, day)), full_file)
                shutil.move(os.path.join(self.nc_folder,get_radial_name(site, day)),  full_file)
                shutil.rmtree(self.nc_folder)
                os.makedirs(self.nc_folder)



def main(data_folder):
    ruv2nc = Ruv2Nc(data_folder)
    thredds = FolderTree(ruv2nc.root_folder)
    ruv2nc.check_nc_files(thredds)


if __name__ == '__main__':
    data_folder = r'../datos/radar'


    main(data_folder)