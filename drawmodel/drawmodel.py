from download_models.download_hydro import DownloadModels
from drawmap.src import drawmap
from common import read_input
import os
from datetime import datetime,timedelta


def draw_model(inputs):
    dates = inputs['dates']
    for date in dates:
        inputs['date_ini'] = date
        file_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(-1)

        download_app = DownloadModels(inputs)
        download_app.download_by_dates()
        print(download_app.get_filename(file_date))

        inputs_draw = drawmap.read_inputs('drawmap.json')
        inputs_draw['file_in'] = download_app.get_filename(file_date)
        date_pieces = date.split("-")
        date_dir = date_pieces[0] + date_pieces[1] + date_pieces[2]
        path_out = os.path.join(inputs_draw['path_out'], date_dir)
        os.makedirs(path_out) if not os.path.exists(path_out) else None
        inputs_draw['path_out'] = path_out

        drawmap.draw_map_24(inputs_draw)


if __name__ == '__main__':
    # input
    input_keys = ['type_url', 'name_grid', 'dates', 'days', 'path_out']
    inputs = read_input('download_models.json', input_keys)
    # end input
    draw_model(inputs)
