# Cognitive Impairment Diagnosis Spark Application
#### by Wei Shi

The Rosmap gene expression profile data file was not uploaded due to its size,
you can find it here:
https://www.synapse.org/#!Synapse:syn2580853/wiki/409853

The gene cluster origin file is available in "csv/" folder, it is extracted and
and transformed from BioGRID data available at:
https://thebiogrid.org/download.php

The detailed report is in ```doc/report.pdf```

For diagnosing patients with or without Alzheimer's Disease, given their
gene expression profile, using gradient boosted trees and top-k clusters of
Entrez IDs selected by t-test scores between samples of patients diagnosed with
Alzheimer's Disease versus those who are not.

## Top-K T-test Scores
This section explains how to run the top_k.py module to calculate top-k
t-test scores between clusters of Entrez IDs that produce the largest
t-test score magnitudes between two means of patients labeled with
Alzheimer's Disease (AD) and those with no cognitive impairment (NCI).

The calculations for student t-test use population standard deviation. Project
report is ```doc/report.pdf```.


**1**. Create an Amazon Web Service (AWS) account:

http://docs.aws.amazon.com/AmazonSimpleDB/latest/DeveloperGuide/AboutAWSAccounts.html


**2**. Create EC2 key-pair:

http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair


**3**. Install AWS command line interface (CLI):

http://docs.aws.amazon.com/cli/latest/userguide/awscli-install-linux.html

Make sure to choose your OS from the left navigator.

**4**. Configure your AWS CLI:

http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html

**5**. Create IAM group:

  http://docs.aws.amazon.com/IAM/latest/UserGuide/id_groups_create.html

**6**. Attach group policy:

  http://docs.aws.amazon.com/IAM/latest/UserGuide/id_groups_manage_attach-policy.html

  Policies required:
  ```
  AmazonEC2FullAccess
  AmazonElasticMapReduceFullAccess
  ```

**7**. Create IAM user and add that user to group created in step 5:

  http://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html

**8**. Create cluster:
  ```
  aws emr create-cluster \
  --name <1> \
  --release-label emr-5.5.0 \
  --applications Name=Spark Name=Hadoop \
  --ec2-attributes KeyName=<2> \
  --instance-type m3.xlarge \
  --instance-count <3> \
  --configurations <4> \
  --use-default-roles
  ```
  ```
  <1>: the name of your cluster.
  <2>: the Ec2 key-pair created in step 2.
  <3>: number of machines or instances to set up.
  <4>: configuration of Spark variables, you can use the one provided in
  "aws/config.json", which has Spark's dynamic allocation turned off and has
  PySpark use Python3.
  ```
  The above command outputs JSON format of your cluster ID, which is used for
  different things related to your AWS EMR cluster, such as:

  - Describing cluster details:
  ```
  aws emr describe-cluster --cluster-id j-35C7A973OSSRQ
  ```

  - Adding more machines to your cluster.
  ```
  aws emr add-instance-groups --cluster-id j-35C7A973OSSRQ --instance-groups InstanceCount=1,InstanceGroupType=core,InstanceType=m3.xlarge
  ```

http://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-launch.html

Example:
  ```
  aws emr create-cluster \
  --name "Alzheimer-gene-distributed-computing" \
  --release-label emr-5.5.0 \
  --applications Name=Spark Name=Hadoop \
  --ec2-attributes KeyName=pandora-id_rsa \
  --instance-type m3.xlarge \
  --instance-count 3 \
  --configurations file:///home/hduser/Documents/spark/aws/config.json \
  --use-default-roles
  ```

The command above creates a cluster named "Alzheimer-gene-distributed-computing"
with 3 instances, 1 master instance and 2 core instances of type m3.xlarge, using key called
"pandora-id_rsa", which is the key name I set up on AWS. The configuration
file is given in local file path:

  ```file:///home/hduser/Documents/spark/aws/config.json```

**9**. Copy Rosmap and gene cluster files to master node.
  - If your files are in Amazon S3. SSH into AWS EMR master node and do:
    ```
    aws s3 cp <bucket URL> <master node local URL>
    ```
    ```
    <bucket URL>: the s3://... URL to your data file.
    <local URL>: the HDFS path to save your data (HDFS host header not required).
    ```

  - If your files are in your local machine, do:
    ```
    scp -i <keypair> <local file> hadoop@<master node address>:~/<file to paste>
    ```

    The -i <keypair> flag is not required if you
    if you're using ssh-agent with your public keys set:
    ```
    scp <local file> hadoop@<master node address>:~/<file to paste>
    ```

**10**. SSH into master node and run spark-submit.
  ```
  spark-submit --deploy-mode cluster <1> <2> <3> <4> <5>
  ```
  **IMPORTANT**: all files must be on the master node (see step 6).
  ```
  <1>: the spark application.
  <2>: input Rosmap file path.
  <3>: input gene cluster file path.
  <4>: input top-k t-score value of each cluster to select.
  <5>: the output file. The first line is time elapsed to calculate top-k
  t-scores, and the rest have the schema:
  (clusterID, t-score, ad mean, nci mean, ad pop. std., nci pop. std.)
  ```

**11**. Add spark step **(OPTIONAL)**:
  ```
  aws emr add-steps \
  --cluster-id j-<1> \
  --steps Type=spark,Name=MyApp,Args=[--deploy-mode,cluster,--conf,spark.yarn.submit.waitAppCompletion=false,s3://<2>/<3>,s3://<2>/<4>,s3://<2>/<5>,<6>,s3://<2>/<7>/], \
  ActionOnFailure=CONTINUE
  ```
  ```
  <1>: id of your cluster received from step 3.
  <2>: your Amazon s3 bucket.
  <3>: the spark application file name in your Amazon S3 bucket.
  <4>: the rosmap patient gene expression file.
  <5>: the gene cluster file.
  <6>: top-k clusters of gene values with largest t-test scores.
  <7>: output folder of resulting computations.
  ```
  http://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-submit-step.html


**12**. Terminate the cluster **(OPTIONAL)**:
  ```
  aws emr terminate-clusters --cluster-ids j-XXXXXXXXXXXX
  ```

  http://docs.aws.amazon.com/emr/latest/ManagementGuide/UsingEMR_TerminateJobFlow.html


**13**. Access YARN web GUI through FoxyProxy **(OPTIONAL)**:
  - Download FoxyProxy add-on for your web browser. You may have to search for
    it in your browser's add-on website.
  - Follow the instructions below to configure the FoxyProxy add-on:

    http://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-connect-master-node-proxy.html

    A copy of FoxyProxy configuration file is in "aws/foxyproxy-settings.xml"
    for your convenience.

  - SSH from the terminal, e.g.:

    ```ssh -ND 8157 hadoop@ec2-34-224-26-209.compute-1.amazonaws.com```

  - Enter the address into your browser with the FoxyProxy add-on, e.g.:

    ```http://ec2-34-224-26-209.compute-1.amazonaws.com:8088/cluster```

## Diagnosis (Machine Learning)
This part of the application diagnoses patients using top-k t-test score gene clusters
from the top-k cluster part of this project as features using Gradient Boosted Trees
(GBT). The workflow is:
  - Get libSVM file from Rosmap file (labeled_to_libSVM.py).
  - Train the model on the libSVM file from step 1 (train_model.py).
  - Use GBT from step 2 to diagnose a patient CSV file (diagnose.py).
  - **(OPTIONAL)** calculate GBT k-fold cross validation (gbt_cross_validate.py).

**1**. If you don't have a libSVM file of the AD and NCI patients, run
  labeled_to_libSVM.py on the Rosmap file. This will automatically extra AD and NCI patients and
  convert the output to libSVM format.

  ```
  Usage: labeled_to_libSVM.py <1> <2> <3> <4>
  ```
  ```
  <1>: input Rosmap patient gene expression file.
  <2>: input gene cluster file.
  <3>: input top-k cluster feature file.
  <4>: output libSVM file.
  ```

**2**. To train a model:
  ```
  Usage: train_model.py <1> <2> <3>
  ```
  ```
  <1>: input labeled libSVM file.
  <2>: input number of tree iterations.
  <3>: output of model.
  ```

**3**. To diagnose a patient, run diagnose.py:
  ```
  Usage: diagnose.py <1> <2> <3> <4> <5>
  ```
  ```
  <1>: input gradient boosted tree model.
  <2>: input patient gene expression file (like Rosmap but without diagnosis)
  <3>: input gene cluster file.
  <4>: input top-k cluster features file.
  <5>: output patient diagnosis file with schema (patient ID, 0.0 or 1.0), where
    0.0 means a prediction of no cognitive disease and 1.0 means the patient has
    been predicted to have Alzheimer's Disease.
  ```
  **IMPORTANT**: schema of input file of patients gene expression profiles to
    diagnose: (patient ID, g1, g2, g3, ...), where g1, g2, g3, ... are gene
    expression values for the corresponding Entrez ID column.


**4**. To do k-fold cross validation:
  ```
  Usage: gbt_cross_validate.py <1> <2> <3>
  ```
  ```
  <1>: input labeled libSVM file.
  <2>: input integer k folds.
  <3>: input integer number of tree iterations.
  ```

**5**. Extract AD and NCI patients from Rosmap file for testing **(OPTIONAL)**:
  ```
  Usage: ad_nci_processor.py <1> <2> <3>
  ```
  ```
  <1>: input Rosmap file.
  <2>: output AD patients file (without diagnosis).
  <3>: output NCI patients file (without diagnosis).
  ```
