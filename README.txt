Cognitive Impairment Diagnosis Spark Application
by Wei Shi

The detailed report is in "doc/report.pdf"

For diagnosing patients with or without Alzheimer's Disease, given their
gene expression profile, using gradient boosted trees and top-k clusters of
Entrez IDs selected by t-test scores between samples of patients diagnosed with
Alzheimer's Disease versus those who are not.

TOP-K
This section deals with how to run the main.py module to calculate top-k
clusters of Entrez IDs that produce the largest t-test score magnitudes.

The calculations for student t-test use population standard deviation. Project
report is "doc/report.pdf".


1. Create an Amazon Web Service (AWS) account:

  http://docs.aws.amazon.com/AmazonSimpleDB/latest/DeveloperGuide/AboutAWSAccounts.html


2. Create EC2 key-pair:

  http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair


3. Install AWS command line interface (CLI):

  http://docs.aws.amazon.com/cli/latest/userguide/awscli-install-linux.html

  Make sure to choose your OS from the left navigator.


4. Configure your AWS CLI:

  http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html


5. Create cluster:
  aws emr create-cluster \
  --name <cluster_name> \
  --release-label emr-5.5.0 \
  --applications Name=Spark Name=Hadoop \
  --ec2-attributes KeyName=<ec2_key> \
  --instance-type m3.xlarge \
  --instance-count <number_of_instances> \
  --configurations <path_to_configurations> \
  --use-default-roles

  The above command outputs JSON format of your cluster ID, which is used for
  different things related to your AWS EMR cluster, such as:
    1. Describing cluster details:
    aws emr describe-cluster --cluster-id j-35C7A973OSSRQ

    2. Adding more machines to your cluster.
    aws emr add-instance-groups --cluster-id j-35C7A973OSSRQ --instance-groups InstanceCount=1,InstanceGroupType=core,InstanceType=m3.xlarge


  <cluster_name>: the name of your cluster.
  <ec2_key>: the Ec2 key-pair created in step 2.
  <path_to_your_configurations>: file path of your configuration JSON file. Can
    be in local directory or on Amazon S3.
  <number_of_instances>: number of machines or instances to set up.
  <path_to_configurations>: configuration of Spark variables, you can use the
    one provided in the "aws/" directory named "config.json", which has Spark's
    dynamic allocation turned off and has PySpark use Python3.

  For more information visit:

  http://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-launch.html

  Example:
  aws emr create-cluster \
  --name "Alzheimer-distributed-computing" \
  --release-label emr-5.5.0 \
  --applications Name=Spark Name=Hadoop \
  --ec2-attributes KeyName=pandora-id_rsa \
  --instance-type m3.xlarge \
  --instance-count 3 \
  --configurations file:///home/hduser/Documents/spark/aws/config.json \
  --use-default-roles

  The command above creates a cluster named "genex" with 3 instances, 1 master
  instance and 2 core instances of type m3.xlarge, using key called
  "pandora-id_rsa", which is the key name I set up on AWS. The configuration
  file is given in local file path:

    file:///home/hduser/Documents/spark/aws/config.json


6. Copy Rosmap and gene cluster files to master node.
  - If your files are in Amazon S3. SSH into AWS EMR master node and do:
    aws s3 cp <bucket URL> <master node local URL>

    <bucket URL>: the s3://... URL to your data file.
    <local URL>: the HDFS path to save your data (HDFS host header not required).

  - If your files are in your local machine, do:
    scp -i <keypair> <local file> hadoop@<master node address>:~/<file to paste>

    The -i <keypair> flag is not required if you
    if you're using ssh-agent with your public keys set:
    scp <local file> hadoop@<master node address>:~/<file to paste>


7. SSH into master node and run spark-submit.
  spark-submit --deploy-mode cluster <spark_app> <rosmap> <gene_cluster> <k> <output>

  IMPORTANT: all files must be on the master node (see step 6).
  <spark_app>: the spark application.
  <rosmap>: the Rosmap file.
  <gene_cluster>: the gene cluster file.
  <k>: the top-k t-score value of each cluster to select.
  <output>: the output file. The first line is time elapsed to calculate top-k
    t-scores, and the rest have the schema:
    (clusterID, t-score, ad mean, nci mean, ad pop. std., nci pop. std.)


8. Add spark step (OPTIONAL):
  aws emr add-steps \
  --cluster-id j-<cluster_ID> \
  --steps Type=spark,Name=MyApp,Args=[--deploy-mode,cluster,--conf,spark.yarn.submit.waitAppCompletion=false,s3://<bucket>/<spark_app>,s3://<bucket>/<rosmap>,s3://<bucket>/<gene_cluster>,<n>,s3://<bucket>/<rdd_output>/], \
  ActionOnFailure=CONTINUE

  <cluster_ID>: id of your cluster received from step 3.
  <bucket>: your Amazon s3 bucket.
  <spark_app>: the spark application file name in your Amazon S3 bucket.
  <rosmap>: the rosmap patient gene expression file.
  <gene_cluster>: the gene cluster file.
  <k>: top-k clusters of gene values with largest t-test scores.
  <rdd_output>: output folder of resulting computations.

  For more information, visit:

  http://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-submit-step.html


9. Terminate the cluster (OPTIONAL):
  aws emr terminate-clusters --cluster-ids j-XXXXXXXXXXXX

  http://docs.aws.amazon.com/emr/latest/ManagementGuide/UsingEMR_TerminateJobFlow.html


10. Access YARN web GUI through FoxyProxy (OPTIONAL):
  a. Download FoxyProxy add-on for your web browser. You may have to search for
    it in your browser's add-on website.
  b. Follow the instructions below to configure the FoxyProxy add-on:

    http://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-connect-master-node-proxy.html

    A copy of FoxyProxy configuration file is in "aws/foxyproxy-settings.xml"
    for your convenience.

  c. SSH from the terminal, e.g.:
    ssh -ND 8157 hadoop@ec2-34-224-26-209.compute-1.amazonaws.com
  d. Enter the address into your browser with the FoxyProxy add-on, e.g.:

    http://ec2-34-224-26-209.compute-1.amazonaws.com:8088/cluster


DIAGNOSIS (Machine Learning)
This part of the application diagnoses patients using top-k gene clusters from
the top-k cluster part of this project as features using Gradient Boosted Trees
(GBT). The workflow is:
  1. Get libSVM file from Rosmap file (labeled_to_libSVM.py).
  2. Train the model on the libSVM file from step 1 (train_model.py).
  3. Use GBT from step 2 to diagnose a patient CSV file (diagnose.py).
  4. (OPTIONAL) calculate GBT k-fold cross validation (gbt_cross_validate.py).


1. If you don't have a libSVM file of the AD and NCI patients, run
  labeled_to_libSVM.py on the Rosmap file. This will automatically extra AD and NCI patients and
  convert the output to libSVM format.

  Usage: to_libSVM.py <1> <2> <3> <4>
  <1>: input Rosmap patient gene expression file.
  <2>: input gene cluster file.
  <3>: input top-k cluster feature file.
  <4>: output libSVM file.


2. To train a model:
  Usage: train_model.py <1> <2> <3>
  <1>: input labeled libSVM file.
  <2>: input number of tree iterations.
  <3>: output of model.


3. To diagnose a patient, run diagnose.py:
  Usage: diagnose.py <1> <2> <3> <4> <5>
  <1>: input gradient boosted tree model.
  <2>: input patient gene expression file (like Rosmap but without diagnosis)
  <3>: input gene cluster file.
  <4>: input top-k cluster features file.
  <5>: output patient diagnosis file with schema (patient ID, 0.0 or 1.0), where
    0.0 means a prediction of no cognitive disease and 1.0 means the patient has
    been predicted to have Alzheimer's Disease.

  IMPORTANT: schema of input file of patients gene expression profiles to
    diagnose: (patient ID, g1, g2, g3, ...), where g1, g2, g3, ... are gene
    expression values for the corresponding Entrez ID column.


4. To do k-fold cross validation:
  Usage: gbt_cross_validate.py <1> <2> <3>
  <1>: input labeled libSVM file.
  <2>: input integer k folds.
  <3>: input integer number of tree iterations.


5. Extract AD and NCI patients from Rosmap file for testing (OPTIONAL):
  Usage: ad_nci_processor.py <1> <2> <3>
  <1>: input Rosmap file.
  <2>: output AD patients file (without diagnosis).
  <3>: output NCI patients file (without diagnosis).
