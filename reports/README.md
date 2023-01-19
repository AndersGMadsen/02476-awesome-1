---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [x] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 1

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s194272, s194260

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used a UNET where the encoder was resnet inspired from the third party frame work Pytorch Image Models. The decoder was found elsewhere as the framework did not offer many selections of segmentation model. Honestly, using a third party framework was a little vit limiting. Obviouly, it assured quality regarding the model and it is always great to implement new model in different frameworks. However, as our starting point was the data set, we did spend some time finding a model from the recommended frameworks that worked for our setting. It was great working with a third party set up in this project.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

In our project, we used a requirements file to keep track of dependencies. The list of dependencies can be auto-generated calling pip install -r requirements.txt or by make requirements. There is also a requirements_test.txt which is used for packages only used for testing purposes. One could easily get a complete copy of the working environment by running those commands. At some point, we also had a requirements file for the api so it is important to have a good setup and workflow to keep track of everything which we did manage to do. testing that all dependencies work with github action was a great tool to learn.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

From the cookiecutter template we have filled out the almost all folders. We did not really use the docs or the references folders in our project. We have added a fastapi folder that contains code for running our locally hosted app. We also added a tests folder that contains pytest tests. For experiments, we have the make_train script as well as the configuration files. The cookie cutter template is really nice and something I think we will keep using. We also did not use the visualisation folder a lot. Normally, we develop a lot in notebooks and it is nice to keep them separate from the actual code.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We used some of the build in methods for assuring data quality and good practice like pep8, isort and linting etc. Most of the rules were implemented to run implicitly, that is, running automatically. Good coding practises really help structure work especially when more people are working on it - which is the case in larger projects.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we implemented about 10 tests in four categories and we are primarily testing the data structures and the model as these are the most critical parts of our application but also some the preprecess functions are tested. Funnily enough, we did not find the actual tests running as usual as the process of implementing them. Obviouly, erros will erise and when that time comes, it is very useful to have tests. We used pytest.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of code is about 85%, which includes all our source code. We do not have 00% coverage of our code and you can reason that the number in itself does not tell a whole lot about the code. The coverage depends on the tests. What we also found was that the process of writing the unittests was more giving than the fact that they were running. The coverage only spits out the number of lines that are run and while it is good to know that lines run without errors, there are many things about the structures of the data and the model that the coverage does not tell you anything about. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

No, we did not really use branches and pull requsts. Generally, they can really help structure the workflow but we were only two people and did a lot together and we did fine without it. If the project had been slightly more substantial, we definitely would have made branches and PR but as we were working so closely and almost everything, we made it work without. The only use of branches was that we created a dev branch to push to after we implemented github actions such that we did not have run all tests everytime we pushed something (which in periods was quite often). If we had used branches, it would probably have been task related and not person related as suggested. 

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we dod use DVC for managing data in our project. There was only one version of the data ever so we did not really use it for version control. However, connecting DVC to google drive and pulling that fro there was very useful and a nice way of sharing data and making it accesible. In a real case scenario, with pictures coming in all the time, DVC would be really great and very  benficial for keepong track of data versions and so on. It is without a doubt etremely beneficial to know exactly what data your model is training on and DVC is a great tool for keeping track of that ecaxtly. In the end, in our pipeline, you do have to pull data using DVC. 

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We have organised our CI in one single file and use github actions and pytest to keep track of it. We use this to get caches, dependencies, linting, (not getting data because the data files were too large to be loaded on github), and unittests. We test for two different operating systems (linux and Mac - Windows was difficult to get up and running) with two version of python (3.9 and 3.10). We could potentially have tested mny other things but this already took 5 minutes. As mentioned, to work around doiing github actions at every push, we pushed to a test branch and then merged into main. we did look at implementing cont. ML as well but even though we did some priliminary tests, we did not include CML in the finished projects both because the data was too large and the trained model as well was huge (1.2GB). We are a little sorry that data was such a limiting factor in this project and but this must also be the case in real world project. Then you just buy more storage and github minutes, I guess.
      
      
      
## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

Our experiemnt is quite easy to run. You pull the data from google drive, you process the data using our make_dataset.py file, and then you can train the model on the processed data with train_model.py. This will take some time (even with GPUs) as the images are large and they are many (9000 train images).

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

first of all, everything is logged. Traning data is version controlled and we used config files. When we train a model, a checkpooint is saved. When an experiment is run, the inference is saved. Everything is kept track of (also using wandb). To reproduce an experiment, one would have to pull the data from drive, process the data using make_dataset.py, train the model using train_model.py and create an inference.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we developed several images: one training and inference deployment. For example to run the training docker image: `docker run trainer:latest`. Link to docker file: <weblink>*. However, we moved the api code into the project folder and it seemed silly to have a dockerfile inside a docker file so in the end, we did not use all the images as maybe you could have.  We also spent a lot of time on docker and it was defnitely one of the most annoying and challenging things tht we worked with in this course. However, we do see the value of images and it is something we will use in the future. 

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

While doing a lot of development, the good old print statements were used a lot but when deploying some of the larger scripts visual studios's build in debugger was of much help. We did some profiling doing the development of cod ebut it was not a lot as the code was already perfect.. just kidding, profiling is definitely very useful but in this project is way maybe not very relevant as worked it out using other methods. still, a quite useful tool that we had not previously worked with. Still, debugging will probably always be depedent on the person - everyone does things differently.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer: 

We tried out the various options. We used engine for training. However, the training was very slow and using GPUs was quite expensive so we ended up training on the DTU server in the end. We used bucket for data but again, google drive also worked well. we used the container registry for docker images. Here, we had a lot of problems with authentications. We used Google Run to deploy the model.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

For deployment, we wrapped our model into an application using fast api and streamlit for the ui. We tried locally serving the model which seem to work. We did not get to serving it in the cloud but that could quite easily be done so that a user could invoke the service. 

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

No, we played around with monitoring the model and testing data drifting and such. However, it was a bit trickier to access our images than in the case of the iris data set. We tried out different methods including a package called deepcheck which would corrupt the images in different ways and see how robust the model was to corrupted images. For monitoring, we could have used google services to monitor the model and the application. It wouls be cool to see how the model beaves over time as well as how sensitive it is to new data (data drifting). So, we would not say we didn't manage it, but is was not evident that it was necissary in our case.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

s194272 used only a few for the MNIST testing and s194260 used almost all on an expensive GPU cluster for a few days. In total about 80 spent. 

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

This was in some ways an easy project to make and in some ways incredibly difficult. In this course, we were presented with so many great tools for doing machine learning in practise and many of the things have we have spent so much time on in earlier project were know tough how to automise. For example, removing biolerplate code with pytorch lightning and using GPUs easily with config files. However, at times, it was also quite difficult to pick and choose between the many tools - which were useful and which were just cool and great to know? (most are useful if not relevant)! One of the things we spent some time on in this project was data management. Managing and maintaining large amounts of data is always difficult. This includes cleaning, preprocessing, and storing data in a way that is easily accessible for both training and deployment. Additionally, ensuring the data is of high quality and is consistent across different stages of the pipeline is crucial for the success of the project. We also spent A LOT of times on making docker work and did choose to make some compormises along the way. We also realised that collaboration and communication is very important for succcesful MLOps. Wokring only two people we had things relatively under control but it can be easy to mess things up when you are more people working together. 

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

This project was made in close collaboration between the two members and all members controbuted to code and structuring of the project. If we have to delegate, s194272 was in charge of data processing, a lot of testing, and the API whereas s194260 was in charge of training the model, configs and building dockers images. 
