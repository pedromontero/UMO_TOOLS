import os
import getradarfiles
import radarhf_waves
from datetime import date


def waves2db(data_folder):
    stations = ['SILL',  'PRIO']
    today = date.today()

    for station in stations:
        path = os.path.join(data_folder, 'radarhf_tmp', 'wls', station)
        today_for_file = today.strftime('%Y_%m_01')
        filename = f'WVLM_{station}_{today_for_file}_0000.wls'
        radarhf_waves.wave2db(station, path, filename)


if __name__ == '__main__':
    root = r'../datos/'
    getradarfiles.get_waves_files(root)
    waves2db(root)
