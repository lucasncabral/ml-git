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

#### Create the people dataset


![dataset](./people_faces/people_faces.jpg)

```python
!ml-git datasets create peopleFaces --categories="computer-vision, images" --bucket-name=faces_bucket --mutability=strict --import='people_faces' --unzip
```

#### We can now proceed with the necessary steps to send the new data to storage.

```python
api.add(entity, entity_name_people, bumpversion=True)
```

Commit the changes

```python
# Custom commit message
message = 'Commit example'

api.commit(entity, entity_name_people, message)
```

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

As we have not yet uploaded any version of our dataset, the bucket is empty.

#### Pushing the people dataset

```python
api.push(entity, entity_name_people)
```

#### Amount of files in the buket after pushing the people dataset

```python
get_bucket_files_count()
```

After sending the data, we can observe the presence of 20 blobs related to the 10 images that were versioned. In this case, two blobs were added for each image in our dataset.

#### Create the famous dataset

Let's create our second dataset that has some images equals to the first dataset.


![dataset](famous_faces/famous_faces.jpg)

```python
!ml-git datasets create famousFaces --categories="computer-vision, images" --bucket-name=faces_bucket --mutability=strict --import='famous_faces' --unzip
```

#### We can now proceed with the necessary steps to send the new data to storage.

```python
api.add(entity, entity_name_famous, bumpversion=True)
```

Commit the changes

```python
# Custom commit message
message = 'Commit example'

api.commit(entity, entity_name_famous, message)
```

And finally, sending the data

```python
api.push(entity, entity_name_famous)
```

#### Amount of files in the buket after pushing the famous dataset

```python
get_bucket_files_count()
```

As you can see, only 4 blobs were added to our bucket. Of the set of 7 images, only 2 images were different from the other dataset, so ml-git can optimize storage by adding blobs related only to these new images.
