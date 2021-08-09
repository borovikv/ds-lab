# ds-lab

## A cookiecutter for a complete out of the box Data Science projects environment in Docker.

### A walkthrough on getting everything up and running:
1. Build docker file using:
```shell
make docker-build
```
2. Prepare project directories, for example for kaggle-airbnb:
```shell
python manage.py kaggle-airbnb
```
The above creates full folder structure (data, notebooks, models, etc) in projects folder within the kaggle-airbnb sub-folder

3. Run ds-lab docker using:
```shell
docker-compose up
```
This launches the docker image. In console you will have a message saying how to access the jupyterlab, analogous to the one below:
```
ds-lab     | [I 2021-08-09 07:31:44.878 ServerApp]  or http://127.0.0.1:8888/lab?token=601131b246b16b7da00dc8179bb53c6e560fa71705402a2a
```
Click on the link above, this will open the jupyterlab in your browser.

### Notes: 
This project is setup so that by default the projects folder in docker is mounted to the ds-lab/projects folder on your machine. The work folder is located only on docker instance, and all data is lost with its re-instanciation. 