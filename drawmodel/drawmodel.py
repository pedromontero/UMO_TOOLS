from download_models import download_hydro
from common import read_input


def draw_model(inputs):
    dates = inputs['dates']
    for date in dates:
        inputs['date_ini'] = date
        download_hydro.main(inputs)



if __name__ == '__main__':
    # input
    input_keys = ['type_url', 'name_grid', 'dates', 'days', 'path_out']
    inputs = read_input('download_models.json', input_keys)
    # end input
    draw_model(inputs)
