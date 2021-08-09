import glob
import os
import sys

import boto3
from botocore.exceptions import ClientError

project_structure = [
    'data/original',  # The original, immutable data dump.
    'data/processed',  # The final, canonical data sets for modeling.
    'data/results',  # The results of your project work.
    'models',  # Trained and serialized models, model predictions, or model summaries.
    'notebooks',  # Jupyter notebooks.
    'reports',  # Generated analysis as HTML, PDF, LaTeX, etc.
    'sql',  # queries.
]

BASE_DIR = os.path.dirname(__file__)
BUCKET = ''
S3_DS_LAB_FOLDER = 'ds-lab-datalab'


def start_project(project_name):
    for path in project_structure:
        os.makedirs(os.path.join(BASE_DIR, 'projects', project_name, path), exist_ok=True)


def pull_data(project_name):
    project_path = get_local_project_path(project_name)
    s3 = session().resource('s3')
    for file in list_files(s3, project_name):
        data = read_from_s3(s3, file)
        file_path = file.replace(get_s3_project_location(project_name), '')
        with open(os.path.join(project_path, file_path), 'wb') as f:
            f.write(data)


def get_local_project_path(project_name):
    return os.path.join(BASE_DIR, 'projects', project_name)


def read_from_s3(s3, key):
    try:
        response = s3.Object(BUCKET, key).get()
        return response['Body'].read()
    except ClientError as ex:
        if ex.response['Error']['Code'] != 'NoSuchKey':
            raise ex


def write_to_s3(s3, key, body):
    s3.Bucket(BUCKET).put_object(Key=key, Body=body)


def session():
    return boto3.Session(profile_name='ellation')


def list_files(s3, project_name):
    key = get_s3_project_location(project_name)
    return [o.key for o in s3.Bucket(BUCKET).objects.filter(Prefix=key)]


def get_s3_project_location(project_name):
    return f'{S3_DS_LAB_FOLDER}/{project_name}/'


def push_data(project_name):
    s3 = session().resource('s3')
    project_path = get_local_project_path(project_name)
    folders_to_sync = ['data', 'models', 'reports']
    for folder in folders_to_sync:
        pathname = f'{project_path}/{folder}/*'
        for file in glob.glob(pathname):
            with open(file, 'rb') as f:
                data = f.read()
            key = get_s3_project_location(project_name) + file.replace(project_path + '/', '')
            write_to_s3(s3, key, body=data)


commands = {
    'start_project': start_project,
    'pull_data': pull_data,
    'push_data': push_data,
}

if __name__ == '__main__':
    command = sys.argv[1]
    name = sys.argv[2]
    commands[command](project_name=name)
