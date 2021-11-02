import os
from datetime import datetime, timedelta


def get_radial_name(site, day):
    day_string = day.strftime('%Y_%m_%d_%H%M')
    return f'HFR-Galicia-{site}_{day_string}.nc'


class FolderTree:

    SETS = ['Totals', 'Radials']
    VERSION = '22'
    STATIONS_BY_SYSTEM = {
        'Galicia': ['PRIO', 'VILA', 'FIST', 'SILL', 'LPRO']
    }

    def __init__(self, root, system='Galicia'):
        self.root = root
        self.system = system

    def get_radial_folder(self, station: str, day: datetime) -> str:
        station_folder = 'NRT' + station + self.VERSION
        year_folder = day.strftime("%Y")
        month_folder = day.strftime("%Y_%m")
        day_folder = day.strftime("%Y_%m_%d")
        return os.path.join(station_folder, year_folder, month_folder, day_folder)

    def make_radial_folder(self, station: str, day: datetime) -> None:
        folder_to_create = os.path.join(self.root, self.get_radial_folder(station, day))
        if not os.path.isdir(folder_to_create):
            os.makedirs(folder_to_create)

    def make_radial_folders_by_system(self, day):
        for station in self.STATIONS_BY_SYSTEM[self.system]:
            self.make_radial_folder(station, day)

    def make_radial_folder_for_days(self, days_before):
        today = datetime.today()
        days = [today - timedelta(days=x) for x in range(days_before)]
        for day in days:
            self.make_radial_folders_by_system(day)

    def get_full_file_nc(self, station: str, day: datetime) -> str:
        folder = self.get_radial_folder(station, day)
        file = get_radial_name(station, day)
        return os.path.join(folder, file)


def main():
    root = os.path.join('', './data/inicio')
    folder_tree = FolderTree(root, 'Galicia')
    folder_tree.make_radial_folder_for_days(34)







if __name__ == '__main__':
    main()