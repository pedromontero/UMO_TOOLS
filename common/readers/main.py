from reader_HDF import ReaderHDF


def main():
    reader = ReaderHDF('../../datos/drawmap/MOHID_Hydrodynamic_Vigo_20180711_0000.hdf5')
    print(reader.longitudes)
    print(reader.latitudes)
    print(reader.get_date(1))
    print(reader.get_date(10))

    print(f'number of longitudes = {reader.n_longitudes} and latitudes = {reader.n_latitudes}')
    reader.close()


if __name__ == "__main__":
    main()
