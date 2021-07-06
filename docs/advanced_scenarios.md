# Additional use cases

As you get familiar with ML-Git, you might feel the necessity of use advanced ML-Git features to solve your problems. Thus, this section aims to provide advanced scenarios and additional use cases.

## Keeping track of a dataset

Often, users can share the same dataset. As the dataset improve, you will need to keep track of the changes. It is very simple to keep check what is new in a shared repository. You just need to navigate to the root of your project. Then, you can execute the command `update`, it will update the metadata repository, allowing visibility of what has been changed since the last update. For example, new ML entity and/or new versions.

```
ml-git repository update
```

In case something new exists in this repository, you will see a output like:
```
INFO - Metadata Manager: Pull [/home/Documents/my-mlgit-project-config/.ml-git/datasets/metadata]
INFO - Metadata Manager: Pull [/home/Documents/my-mlgit-project-config/.ml-git/labels/metadata]
```

Then, you can checkout the new available data.


## Adding special credentials AWS

Depending the project you are working on, you might need to use special credentials to restrict access to your entities (e.g., datases) stored inside a S3/MinIO bucket. The easiest way to configure and use a different credentials for the AWS storage is installing the AWS command-line interface (awscli). First, install the awscli. Then, run the following command:

```
aws configure --profile=mlgit
```

You will need to inform the fields listed below.

```
AWS Access Key ID [None]: your-access-key
AWS Secret Access Key [None]: your-secret-access-key
Default region name [None]: bucket-region
Default output format [None]: json
```

These commands will create the files ~/.aws/credentials and ~/.aws/config.

- Demonstrating AWS Configure
  
[![asciicast](https://asciinema.org/a/371052.svg)](https://asciinema.org/a/371052)

After you have created your special credentions (e.g., mlgit profile)

You can use this profile as parameter to access your storages. Following, you will find an exaple where we attached the profile to the storage mlgit-datasets.

```
ml-git repository storage add mlgit-datasets --credentials=mlgit
```