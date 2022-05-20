# ds-lab

This is a project for a complete out of the box Data Science projects environment in Docker.

### Requirements

1. Docker - see installation instructions [here](https://docs.docker.com/desktop/)
2. git - check in terminal git --version, here is [installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if there is no git.

### A walkthrough on getting everything up and running:
1. Clone project:
    ```shell
    git clone https://github.com/borovikv/ds-lab.git
    ```
2. Chane directory:
    ```shell
    cd ds-lab
    ```

3. Build docker file using:
    ```shell
    make build
    ```
4. Prepare project directories, for example for kaggle-airbnb:
    ```shell
    make project name=Project_Name
    ```
    The above creates full folder structure (data, notebooks, models, etc) in projects folder
    within the **Project_Name** sub-folder inside of `../jupiter-projects` folder.
    

6. Run ds-lab docker using:
    ```shell
    make up
    ```

7. Open Jupyter notebook in your browser run:
    ```shell
    make jupyter
    ```

8. When you are finished with your work or there is need to restart Jupyter, run:
    ```shell
    make down
    ```

### Proposed Project structure
A project created with `make project name=Project_Name` command will have this structure. 
Parent directory is `../jupiter-projects/Project_Name/`
1. **data/original** - The original, immutable data dump.
2. **data/processed** - The final, canonical data sets for modeling.
3. **data/results** - The results of your project work.
4. **models** - Trained and serialized models, model predictions, or model summaries.
5. **notebooks** - Jupyter notebooks.
6. **reports** - Generated analysis as HTML, PDF, LaTeX, etc.
7. **sql** - queries.

### Useful links

1. [Jupyter Notebook Tips, Tricks, and Shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/#:~:text=Select%20Multiple%20Cells%3A,run%20them%20as%20a%20batch.)
2. [SciKit](https://scikit-learn.org/stable/index.html)