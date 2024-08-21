import os

def create_work_dir(workdir):
    # Creating workdir
    if not os.path.isdir(workdir):
        os.makedirs(workdir)