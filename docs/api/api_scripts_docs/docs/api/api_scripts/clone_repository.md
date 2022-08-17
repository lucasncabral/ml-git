---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.7.0 64-bit
    language: python
    name: python3
---

## Cloning an ml-git repository


This notebook describes how to clone an ML-Git repository.


### Notebook state management


If you have already run this notebook or another in this same folder, it is recommended that you perform a state restart by executing the cell below, because previously performed state changes may interfere with the execution of the notebook. Be aware, that procedure does not affect any remote repository.

```python
%cd /api_scripts
!rm -rf ./logs
!rm -rf .ml-git
!rm -rf ./datasets
```

#### To start using the ml-git api we need to import it into our script

```python
from ml_git.api import MLGitAPI
```

#### After importing you can use the api clone method, passing the url of the git repository as a parameter.

```python
repository_url = '/local_ml_git_config_server.git'
api = MLGitAPI()

api.clone(repository_url)
```

#### When the clone is successfully completed, the entities are initialized and ready for use.
