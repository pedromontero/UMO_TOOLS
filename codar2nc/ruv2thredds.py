import os
import re
from datetime import datetime, timedelta
from create_dir_structure import FolderTree





def check_nc_files(thredds, ruv_folder):
    for file in os.listdir(ruv_folder):
        site = re.findall("[A-Z]{4}", file.split('/')[-1])[0]
        day = datetime.strptime('%s%s%s%s' % tuple(re.findall("\d+", file.split('/')[-1])), '%Y%m%d%H%M')

        full_file = thredds.get_full_file_nc(site,day)
        if not check_nc_file(full_file):
            print(f'---------------- voy a crear {full_file}')
            day_hour_before = day + timedelta(hours=-1)
            file_previous_hour = thredds.get_full_file_nc(site, day_hour_before)

            day_hour_after = day + timedelta(hours=+1)
            file_after_hour = thredds.get_full_file_nc(site, day_hour_after)
            print(f'---vou necesitar {file_previous_hour} e {file_after_hour}')

def check_nc_file(full_file):
        if os.path.isfile(full_file):
            print(f'existe {full_file} ')
            return True
        else:
            print(f'No existe {full_file}')
            return False

def main(root_folder, ruv_folder):

    thredds = FolderTree(root_folder)

    print(check_nc_files(thredds, ruv_folder))

if __name__ == '__main__':
    data_folder = r'../datos/radar'
    ruv_folder = os.path.join(data_folder, 'radials')
    root_folder = os.path.join(data_folder, 'thredds')

    main(root_folder, ruv_folder)