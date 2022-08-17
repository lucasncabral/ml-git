## Basic Flow

This notebook describes a basic execution flow of ml-git with its API. There, you will learn how to initialize an ML-Git project, how to perform all the necessary configuration steps and how to version a dataset. We will divide this quick howto into 3 main sections:

##### ml-git repository configuation/intialization 
     This section explains how to initialize and configure a repository for ml-git. 
##### versioning a dataset
    Having a repository initialized, this section explains how to create and upload a dataset to the storage. 
##### downloading a dataset
    This section describes how to download a versioned data set using ml-git.

### Notebook state management

If you have already run this notebook or another in this same folder, it is recommended that you perform a state restart by executing the cell below, because previously performed state changes may interfere with the execution of the notebook. Be aware, that procedure does not affect any remote repository.


```python
%cd /api_scripts
!rm -rf ./logs
!rm -rf .ml-git
!rm -rf ./datasets
```

### 1 - ml-git repository configuation/intialization 

To start using the ml-git api we need to import it into our script


```python
from ml_git.api import MLGitAPI
```

    

Then we must create a new instance of the API class


```python
api = MLGitAPI()
```

To use ml-git, it is necessary to configure storages and remotes that will be used in a project. This configuration can be done through a sequence of commands, or if you already have a git repository with the stored settings, you can run the clone command to import those settings. The following subsections demonstrate how to configure these two forms. 

**Note: You should only perform one of the following two subsections.**

**1.1 Configuring with clone command**

With the clone command all settings will be imported and initialized from the repository that was informed.


```python
repository_url = '/local_ml_git_config_server.git'

api.clone(repository_url)
```

After that, you can skip to section 2 which teaches you how to create a version of a dataset.

**1.2 Configuring from start**

In this section we will consider the scenario of a user who wants to configure their project from scratch. The first step is to define that the directory we are working on will be an ml-git project, for this we execute the following command:


```python
api.init('repository')
```

    INFO - Admin: Initialized empty ml-git repository in /api_scripts/.ml-git


    

After initializing an ml-git project, it is necessary that you inform the remotes and storages that will be used by the entities to be versioned. If you want to better understand why ml-git uses these resources, please take a look at the [architecture and internals documentation](../../mlgit_internals.md).

In this notebook we will configure our ml-git project with a local git repository and a local minio as storage. For this, the following commands are necessary:


```python
remote_url = '/local_server.git/'
bucket_name= 'mlgit'
end_point = 'http://127.0.0.1:9000'

# The type of entity we are working on
entity_type = 'datasets'

api.remote_add(entity_type, remote_url)
api.storage_add(bucket_name, endpoint_url=end_point)
```

    INFO - Admin: Add remote repository [/local_server.git/] for [datasets]
    INFO - Admin: Add storage [s3h://mlgit]


    

Last but not least, initialize the metadata repository


```python
api.init(entity_type)
```

    INFO - Metadata Manager: Metadata init [/local_server.git/] @ [/api_scripts/.ml-git/datasets/metadata]


    

### 2 - versioning a dataset

After the entities have been initialized and are ready for use. We can continue with the process to version our first dataset.

ml-git expects any dataset to be specified under *dataset/* directory of your project and it expects a specification file with the name of the dataset.

To create this specification file for a new entity you must run the following command:


```python
# The entity name we are working on
entity_name = 'dataset-ex'

api.create(entity_type, entity_name, categories=['computer-vision', 'images'], mutability='strict', bucket_name=bucket_name)
```

    INFO - MLGit: Dataset artifact created.


    

Once we create our dataset entity we can add the data to be versioned within the entity's directory. For this, the following code generate a new file in our dataset path.


```python
import os

def create_file(file_name='file'):
    file_path = os.path.join('datasets', 'dataset-ex', 'data', file_name)
    open(file_path, 'a').close()

create_file()
```

    

We can now proceed with the necessary steps to send the new data to storage.


```python
api.add(entity_type, entity_name, bumpversion=True)
```

    INFO - Metadata Manager: Pull [/api_scripts/.ml-git/datasets/metadata]
    INFO - Repository: datasets adding path [/api_scripts/datasets/dataset-ex] to ml-git index
    files: 100%|██████████| 1.00/1.00 [00:00<00:00, 360files/s]


    ⠋ Creating hard links in cache⠙ Creating hard links in cache

    files: 100%|██████████| 1.00/1.00 [00:00<00:00, 4.13kfiles/s]


    

After add the files, you need commit the metadata to the local repository. For this purpose type the following command:


```python
# Custom commit message
message = 'Commit example'

api.commit(entity_type, entity_name, message)
```

    ⠋ Updating index⠙ Updating index Checking removed files⠙ Checking removed files Commit manifest⠙ Commit manifest

    INFO - Metadata Manager: Commit repo[/api_scripts/.ml-git/datasets/metadata] --- file[dataset-ex]


    

Last but not least, ml-git dataset push will update the remote metadata repository just after storing all actual data under management in the specified remote data storage.


```python
api.push(entity_type, entity_name)
```

    files: 100%|██████████| 1.00/1.00 [00:00<00:00, 57.4files/s]


    ⠋ Pushing metadata to the git repository⠙ Pushing metadata to the git repository

As you can observe, ml-git follows very similar workflows as for git.

### 3 - downloading a dataset

Once you have an entity versioned by ml-git, and being within an initialized directory, it is really simple to obtain data from a specific entity. As an example, in this notebook we will checkout an entity that was previously versioned, the mnist. For this, the following command is necessary:


```python
entity_name = 'mnist'

data_path = api.checkout(entity_type, entity_name, version=1)
```

    INFO - Metadata Manager: Pull [/api_scripts/.ml-git/datasets/metadata]
    INFO - Metadata: Performing checkout in tag handwritten__digits__mnist__1
    blobs: 100%|██████████| 2.00/2.00 [00:00<00:00, 186blobs/s]
    chunks: 100%|██████████| 2.00/2.00 [00:00<00:00, 2.24chunks/s]
    files into workspace: 100%|██████████| 2.00/2.00 [00:00<00:00, 12.6files into workspace/s]


    

Getting the data will auto-create a directory structure under dataset directory. That structure *computer-vision/images* is actually coming from the categories defined in the dataset spec file. Doing that way allows for easy download of many datasets in one single ml-git project without creating any conflicts.

Now the user can perform the processes he wants with the data that was downloaded in the workspace.
