import os
import json
from collections import OrderedDict
import paramiko
from stat import S_ISDIR
paramiko.util.log_to_file("paramiko.log")


def get_path_out(path_out):
    os.chdir(r'.\..')
    root = os.getcwd()
    path_out = os.path.join(root, path_out)
    return path_out


def read_connection(input_file):
    try:
        with open(input_file, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        print(f'File not found: {input_file} ')
        if input('Do you want to create one (y/n)?') == 'n':
            quit()


def sftp_walk(sftp, remote_path):

    path = remote_path
    files = []
    folders = []
    for f in sftp.listdir_attr(remote_path):
        if S_ISDIR(f.st_mode):
            folders.append(f.filename)
        else:
            files.append(f.filename)
    if files:
        yield path, files
    for folder in folders:
        new_path = os.path.join(remote_path, folder)
        for x in sftp_walk(new_path):
            yield x


def sftp_get_filenames_by_extension(sftp, remote_path, extension):

    path = remote_path
    files = []
    for f in sftp.listdir_attr(remote_path):
        if not S_ISDIR(f.st_mode):
            if f.filename[-3:] == extension:
                files.append(f.filename)
    if files:
        yield path, files


def ssh_connection_password(path):

    connection_params = read_connection(path)
    transport = paramiko.Transport(connection_params['host'], 22)
    transport.connect(username=connection_params['user'], password=connection_params['password'])
    return transport


def get_stfp(connection_params_path):
    transport = ssh_connection_password(connection_params_path)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp


def get_radar_files(root_dir):

    root_dir = os.path.join(root_dir, 'radarhf_tmp', 'ruv')

    sftp = get_stfp(r'./pass/combine.json')
    stations = ['LPRO', 'SILL', 'VILA', 'PRIO', 'FIST']
    remote_root_path =r'/Codar/SeaSonde/Data/RadialSites/Site_'
    for station in stations:

        remote_path = remote_root_path + station
        print(f'remote path = {remote_path}')

        for path, files in sftp_get_filenames_by_extension(sftp, remote_path, 'ruv'):
            for file in files:
                print(f'Atopei o ficheiro {file} no cartafol {path}')
                if file.split('.')[-1] == 'ruv':
                    remote_file = path + "/" + file
                    local_dir = os.path.join(root_dir, station)
                    if not os.path.exists(local_dir):
                        os.makedirs(local_dir)
                    local_file = os.path.join(local_dir, file)
                    if not os.path.exists(local_file):
                        print(f'Get from {remote_file} to {local_file}')
                        sftp.get(remote_file, local_file)
                    else:
                        print(f'{file} xa está baixado')
                        pass
    sftp.close()

def main():
    data_folder = r'../datos'
    get_radar_files(data_folder)


if __name__ == '__main__':
    main()




