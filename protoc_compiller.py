import subprocess
from os import listdir

protobuf_dir = './object_detection/protos'
protobuf_path_list = [f for f in listdir(protobuf_dir) if f.endswith('.proto')]

for path in protobuf_path_list:
    try:
        subprocess.Popen('protoc {0}/{1} --python_out=.'.format(protobuf_dir, path))
    except Exception as e:
        print('Error: \'protoc\' is not recognized as an internal or external command, operable program or batch file.')
        print('Download and install \'protoc\' for your platform: https://github.com/protocolbuffers/protobuf/releases')
        exit(0)

print('successfully!')
