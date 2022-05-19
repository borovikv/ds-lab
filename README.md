# ds-lab

This is a project for a complete out of the box Data Science projects environment in Docker.

### Requirements

1. Docker - see installation instructions [here](https://docs.docker.com/desktop/)
2. git - check in terminal git --version, here is [installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if there is no git.

### A walkthrough on getting everything up and running:
1. Build docker file using:
```shell
make build
```
2. Prepare project directories, for example for kaggle-airbnb:
```shell
make project name=Project_Name
```
The above creates full folder structure (data, notebooks, models, etc) in projects folder within the kaggle-airbnb sub-folder

3. Run ds-lab docker using:
```shell
make up
```

4. Open Jupyter notebook in your browser run:
```shell
make jupyter
```

5. When you are finished with your work or there is need to restart Jupyter, run:
```shell
make down
```