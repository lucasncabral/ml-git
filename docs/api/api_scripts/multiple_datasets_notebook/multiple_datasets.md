## Storage reuse with multiple datasets

#### This notebook describes how to handle the scenario where the same file is present in more than one dataset.

When the same file is used in multiple datasets, that file will be added to the bucket only once, in order to optimize the space usage in the bucket. To exemplify this use case, two entities will be created: the people entity contains 10 images with faces of people, while famous entity contains 7 images with faces of famous people, being 5 of them also contained in the people entity.

This way, when sending the files to the repository, the 5 images that are being used in the two entities will not be duplicated in the bucket. The two entities will refer to the same image stored in the bucket.

### Notebook state management

If you have already run this notebook or another in this same folder, it is recommended that you perform a state restart by executing the cell below, because previously performed state changes may interfere with the execution of the notebook. Be aware, that procedure does not affect any remote repository.


```python
%cd /api_scripts/multiple_datasets_notebook
!rm -rf ./logs
!rm -rf .ml-git
!rm -rf ./datasets
```

#### To start using the ml-git api we need to import it into our script


```python
from ml_git.api import MLGitAPI
```

    

#### After that, we define some variables that will be used by the notebook


```python
# The type of entity we are working on
entity = 'datasets'

# The entity name we are working on
entity_name_people = 'peopleFaces'

# The entity name we are working on
entity_name_famous = 'famousFaces'
```

    

#### To start, let's take into account that you have a repository with git settings to make the clone. If this is not your scenario, you will need to configure ml-git outside this notebook (At the moment the api does not have the necessary methods to perform this configuration). 

#### Or you can manually configure the repository using the command line, following the steps in the [First Project](https://github.com/HPInc/ml-git/blob/development/docs/first_project.md) documentation.


```python
repository_url = '/local_ml_git_config_server.git'
api = MLGitAPI()

api.clone(repository_url)
```

    INFO - Metadata Manager: Metadata init [/local_server.git/] @ [/api_scripts/multiple_datasets_notebook/tmpfyoxmtjy/mlgit/.ml-git/datasets/metadata]
    INFO - Metadata Manager: Metadata init [/local_server.git/] @ [/api_scripts/multiple_datasets_notebook/tmpfyoxmtjy/mlgit/.ml-git/models/metadata]
    INFO - Metadata Manager: Metadata init [/local_server.git/] @ [/api_scripts/multiple_datasets_notebook/tmpfyoxmtjy/mlgit/.ml-git/labels/metadata]
    INFO - Metadata: Successfully loaded configuration files!


    

#### Create the people dataset

![dataset](./people_faces/people_faces.jpg)


```python
!ml-git datasets create peopleFaces --categories="computer-vision, images" --bucket-name=faces_bucket --mutability=strict --import='people_faces' --unzip
```

    [?25h⠋ Importing files⠙ Importing filesINFO - MLGit: Unzipping files
    INFO - MLGit: Dataset artifact created.
    [?25h

#### We can now proceed with the necessary steps to send the new data to storage.


```python
api.add(entity, entity_name_people, bumpversion=True)
```

    INFO - Metadata Manager: Pull [/api_scripts/multiple_datasets_notebook/.ml-git/datasets/metadata]
    INFO - Repository: datasets adding path [/api_scripts/multiple_datasets_notebook/datasets/people_faces] to ml-git index
    files: 100%|██████████| 10.0/10.0 [00:00<00:00, 646files/s]


    ⠋ Creating hard links in cache⠙ Creating hard links in cache

    files: 100%|██████████| 10.0/10.0 [00:00<00:00, 32.7kfiles/s]


    

Commit the changes


```python
# Custom commit message
message = 'Commit example'

api.commit(entity, entity_name_people, message)
```

    ⠋ Updating index⠙ Updating index Checking removed files⠙ Checking removed files Commit manifest⠙ Commit manifest

    INFO - Metadata Manager: Commit repo[/api_scripts/multiple_datasets_notebook/.ml-git/datasets/metadata] --- file[people_faces]


    

#### As we are using MinIO locally to store the data in the bucket, we were able to check the number of files that are in the local bucket.


```python
import os

def get_bucket_files_count():
  print("Number of files on bucket: " +  str(len(os.listdir('../../data/faces_bucket'))))
```

    

#### Amount of files in the buket before pushing the people dataset


```python
get_bucket_files_count()
```

    Number of files on bucket: 0
    

As we have not yet uploaded any version of our dataset, the bucket is empty.

#### Pushing the people dataset


```python
api.push(entity, entity_name_people)
```

    files: 100%|██████████| 20.0/20.0 [00:00<00:00, 92.3files/s]


    ⠋ Pushing metadata to the git repository⠙ Pushing metadata to the git repository

#### Amount of files in the buket after pushing the people dataset


```python
get_bucket_files_count()
```

    Number of files on bucket: 20
    

After sending the data, we can observe the presence of 20 blobs related to the 10 images that were versioned. In this case, two blobs were added for each image in our dataset.

#### Create the famous dataset

Let's create our second dataset that has some images equals to the first dataset.

![dataset](famous_faces/famous_faces.jpg)


```python
!ml-git datasets create famousFaces --categories="computer-vision, images" --bucket-name=faces_bucket --mutability=strict --import='famous_faces' --unzip
```

    [?25h⠋ Importing files⠙ Importing filesINFO - MLGit: Unzipping files
    INFO - MLGit: Dataset artifact created.
    [?25h

#### We can now proceed with the necessary steps to send the new data to storage.


```python
api.add(entity, entity_name_famous, bumpversion=True)
```

    INFO - Metadata Manager: Pull [/api_scripts/multiple_datasets_notebook/.ml-git/datasets/metadata]
    INFO - Repository: datasets adding path [/api_scripts/multiple_datasets_notebook/datasets/famous_faces] to ml-git index
    files: 100%|██████████| 7.00/7.00 [00:00<00:00, 703files/s]


    ⠋ Creating hard links in cache⠙ Creating hard links in cache

    files: 100%|██████████| 7.00/7.00 [00:00<00:00, 40.2kfiles/s]


    

Commit the changes


```python
# Custom commit message
message = 'Commit example'

api.commit(entity, entity_name_famous, message)
```

    ⠋ Updating index⠙ Updating index Checking removed files⠙ Checking removed files Commit manifest⠙ Commit manifest

    INFO - Metadata Manager: Commit repo[/api_scripts/multiple_datasets_notebook/.ml-git/datasets/metadata] --- file[famous_faces]


    

And finally, sending the data


```python
api.push(entity, entity_name_famous)
```

    files: 100%|██████████| 14.0/14.0 [00:00<00:00, 177files/s]


    ⠋ Pushing metadata to the git repository⠙ Pushing metadata to the git repository

#### Amount of files in the buket after pushing the famous dataset


```python
get_bucket_files_count()
```

    Number of files on bucket: 24
    

As you can see, only 4 blobs were added to our bucket. Of the set of 7 images, only 2 images were different from the other dataset, so ml-git can optimize storage by adding blobs related only to these new images.
