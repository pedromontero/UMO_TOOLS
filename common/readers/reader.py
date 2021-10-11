from abc import ABC, abstractmethod


class Reader(ABC):
    def __init__(self, file):
        self.dataset = self.open(file)
        self.n_longitudes = None
        self.n_latitudes = None
        self.longitudes = self.get_longitudes()
        self.latitudes = self.get_latitudes()

    @abstractmethod
    def open(self, file):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_latitudes(self):
        pass

    @abstractmethod
    def get_longitudes(self):
        pass

    @abstractmethod
    def get_dates(self):
        pass

    @abstractmethod
    def get_date(self, n_time):
        pass

