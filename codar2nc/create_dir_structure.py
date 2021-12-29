import os
from datetime import datetime, timedelta




class FolderTree:

    SETS = ['Totals', 'Radials']
    VERSION = '22'
    STATIONS_BY_SYSTEM = {
        'Galicia': ['PRIO', 'VILA', 'FIST', 'SILL', 'LPRO']
    }

    def __init__(self, root, system='Galicia'):
        self.root = root
        self.system = system

    def get_name(self, local, day):
        day_string = day.strftime('%Y_%m_%d_%H%M')
        return f'HFR-{self.system}-{local}_{day_string}.nc'

    def get_radial_folder(self, station: str, day: datetime) -> str:
        station_folder = 'NRT' + station + self.VERSION
        year_folder = day.strftime("%Y")
        month_folder = day.strftime("%Y_%m")
        day_folder = day.strftime("%Y_%m_%d")
        return os.path.join(station_folder, year_folder, month_folder, day_folder)

    def get_total_folder(self, day: datetime) -> str:
        system_folder = self.system + '_NRT' + self.VERSION
        year_folder = day.strftime("%Y")
        month_folder = day.strftime("%Y_%m")
        day_folder = day.strftime("%Y_%m_%d")
        return os.path.join(system_folder, year_folder, month_folder, day_folder)

    def make_radial_folder(self, station: str, day: datetime) -> None:
        if not self.exist_radial_folder(station, day):
            folder_to_create = os.path.join(self.root, self.get_radial_folder(station, day))
            os.makedirs(folder_to_create)

    def make_total_folder(self, day: datetime) -> None:
        if not self.exist_total_folder( day):
            folder_to_create = os.path.join(self.root, self.get_total_folder(day))
            os.makedirs(folder_to_create)

    def make_radial_folders_by_system(self, day):
        for station in self.STATIONS_BY_SYSTEM[self.system]:
            self.make_radial_folder(station, day)

    def make_radial_folder_for_days(self, days_before):
        today = datetime.today()
        days = [today - timedelta(days=x) for x in range(days_before)]
        for day in days:
            self.make_radial_folders_by_system(day)

    def get_full_radial_file_nc(self, station: str, day: datetime) -> str:
        folder = self.get_radial_folder(station, day)
        file = self.get_name(station, day)
        return os.path.join(folder, file)

    def get_full_total_file_nc(self, day: datetime) -> str:
        folder = self.get_total_folder(day)
        file = self.get_name('Total', day)
        return os.path.join(folder, file)

    def exist_radial_folder(self, station: str, day: datetime) -> bool:
        return os.path.isdir(os.path.join(self.root, self.get_radial_folder(station, day)))

    def exist_total_folder(self, day: datetime) -> bool:
        return os.path.isdir(os.path.join(self.root, self.get_total_folder(day)))


def main():
    root = os.path.join('', './data/inicio')
    folder_tree = FolderTree(root, 'Galicia')
    folder_tree.make_radial_folder_for_days(34)


if __name__ == '__main__':
    main()