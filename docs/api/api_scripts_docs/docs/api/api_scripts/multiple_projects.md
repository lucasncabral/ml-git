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

## Multiple Projects


This notebook describes how to work with multiple projects in the ML-Git API.


### Notebook state management


If you have already run this notebook or another in this same folder, it is recommended that you perform a state restart by executing the cell below, because previously performed state changes may interfere with the execution of the notebook. Be aware, that procedure does not affect any remote repository.

```python
%cd /api_scripts
!rm -rf ./logs
!rm -rf .ml-git
!rm -rf ./datasets
!rm -rf /api_scripts/project_1
!rm -rf /api_scripts/project_2
```

### Using multiples projects


Let's start creating, for each project, a folder to work with.

```python
# importing os module
import os

os.mkdir('project_1')
os.mkdir('project_2')
```

To start using the ML-Git API, we need to import it into our script.

```python
from ml_git.api import MLGitAPI
```

Then we must create a new instance of the API for each project. You can inform the root directory of the project as a parameter. In this scenario, we will be using the relative path to each directory.

```python
project_1 = MLGitAPI(root_path='./project_1')
project_2 = MLGitAPI(root_path='./project_2')
```

We will consider the scenario of a user who wants to configure their projects from scratch. The first step is to define that the directory we are working on will be an ml-git project. To do that, execute the following command:

```python
project_1.init('repository')
project_2.init('repository')
```

We will configure each project with a local git repository and a local MinIO as storage. For this, the following commands are necessary:

```python
remote_url = '/local_server.git/'
bucket_name= 'mlgit'
end_point = 'http://127.0.0.1:9000'

# The type of entity we are working on
entity_type = 'datasets'

project_1.remote_add(entity_type, remote_url)
project_1.storage_add(bucket_name, endpoint_url=end_point)

print('/n')

project_2.remote_add(entity_type, remote_url)
project_2.storage_add(bucket_name, endpoint_url=end_point)
```

After the projects have been initialized we can continue with the process to create new datasets.
To create the specification file for new entities you must run the following commands:

```python
# The entity name we are working on project 1
entity_name_1 = 'dataset-ex-1'
# The entity name we are working on project 2
entity_name_2 = 'dataset-ex-2'

project_1.create(entity_type, entity_name_1, categories=['img'], mutability='strict')
project_2.create(entity_type, entity_name_2, categories=['img'], mutability='strict')
```

In this example, we demonstrated how to work with multiple projects in the ML-Git API. You can use all commands available in the API with this concept of multiple projects, a complete flow of how to version an entity can be found in the [Basic Flow Notebook](./basic_flow.ipynb).
