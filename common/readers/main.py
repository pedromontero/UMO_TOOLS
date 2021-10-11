from reader_HDF import ReaderHDF
from reader_NetCDF import ReaderNetCDF


def main():
    hdf = True
    ncdf = True
    if hdf:
        reader = ReaderHDF('../../datos/drawmap/MOHID_Hydrodynamic_Vigo_20180711_0000.hdf5')
        print(reader.longitudes)
        print(reader.latitudes)
        print(reader.get_date(1))
        print(reader.get_date(10))

        print(f'number of longitudes = {reader.n_longitudes} and latitudes = {reader.n_latitudes}')
        reader.close()

    if ncdf:
        reader = ReaderNetCDF('../../datos/drawmap/wrf_arw_det1km_history_d05_20180711_0000.nc4')

        print(reader.latitudes)
        print(reader.longitudes)
        print(reader.get_date(1))
        print(reader.get_date(10))
        print(f'number of longitudes = {reader.n_longitudes} and latitudes = {reader.n_latitudes}')
        print(reader.get_variable('wind_module', 1))
        reader.close()


if __name__ == "__main__":
    main()