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


def ssh_connection_password(path):

    connection_params = read_connection(r'.\..\datos\pass\combine.json')
    print(connection_params['host'])

    transport = paramiko.Transport(connection_params['host'], 22)
    transport.connect(username=connection_params['user'], password=connection_params['password'])
    return transport


def main():

    connection_params_path = r'.\..\datos\pass\combine.json'
    key_file = r'.\..\datos\pass\id_rsa.pub'
    transport = ssh_connection_password(connection_params_path)
    sftp = paramiko.SFTPClient.from_transport(transport)
    remote_path = '/Codar/SeaSonde/Data/RadialSites/Site_FIST'

    for path, files in sftp_walk(sftp, remote_path):
        for file in files:
            if file.split('.')[-1] == 'ruv':
                remote_file = path + "/" + file
                local_file = os.path.join('../datos/radar/radials', file)
                if not os.path.exists(local_file):
                    print(f'Get from {remote_file} to {local_file}')
                    sftp.get(remote_file, local_file)
                else:
                    print(f'{file} xa está baixado')
    #file_name = 'RDLm_FIST_2021_10_23_0300.ruv'
    #sftp.get(file_name, os.path.join(get_path_out('datos/getradarfiles'), file_name))
    sftp.close()


if __name__ == '__main__':
    main()




