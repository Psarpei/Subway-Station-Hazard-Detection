# Subway Station Hazard Detection, Goethe University Frankfurt (Spring 2020)

## General Information
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

**Instructors:**
* [Prof. Dr. Visvanathan Ramesh](http://www.ccc.cs.uni-frankfurt.de/people/), email: V.Ramesh@em.uni-frankfurt.de
* [Dr. Michael Rammensee](http://www.ccc.cs.uni-frankfurt.de/michaelrammensee/), email: M.Rammensee@em.uni-frankfurt.de

**Institutions:**
  * **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
  * **[AISEL - AI Systems Engineering Lab](http://www.ccc.cs.uni-frankfurt.de/)**
  
**Project team (A-Z):**
* [Pascal Fischer](https://github.com/Psarpei)
* Felix Hoffman
* Edis Kurtanovic
* Martin Ludwig
* [Alen Smajic](https://github.com/alen-smajic/)

## Publications ##
  * **[ResearchGate](https://www.researchgate.net/publication/344830620_Subway_Station_Hazard_Detection)**
  
## Tools ## 
* Python 3
* PyTorch Framework
* Unity3D
* C#
* Blender
* OpenCV

## Project Description ##
<img align="center" width="1000" height="" src="Documentation%20and%20final%20presentation/System%20concept.png">
Recently there was a nationwide scandal about an incident at Frankfurt Central Station in which a boy was pushed onto the railroad tracks in front of an arriving train and lost  his life due to its injuries. However, this incident is not the only one of its kind because such accidents occur from time to time at train stations. At the moment there are no security systems to prevent such accidents, and cameras only exist to provide evidence or clarification after an incident has already happened. In this project, we developed a first prototype of a security system which uses the surveillance cameras at subway stations in combination with the latest deep learning computer vision methods and integrates them into an end-to-end system, which can recognize dangerous situations and initiate measures to prevent further consequences.  Furthermore, we present a 3D subway station simulation, developed in Unity3D, which generates entire train station environments and scenes using an advanced algorithm trough a real data-based distribution of persons.  This simulation is then used to generate training data for the deep learning model. 

### Subway Station Simulation ###
<img align="left" width="1000" height="" src="Result%20images/Simulation%20images/Example%20station%20scenario%201.jpg">
For our simulation we developed 10 different types of subway stations, covering the most common station architectures. The number of platforms varies between 1 and 2, while the number of tracks lies between 1 and 4, for each station. Every station type was manually textured in 5 different variations to get a total of 50 unique station environments. 

<img align="left" width="500" height="" src="Result%20images/Simulation%20images/Example%20station%2011.png">
<img align="left" width="500" height="" src="Result%20images/Simulation%20images/Example%20station%203.png">

Furthermore, we include a variety of different human models as well as station objects like benches, snack machines, stairs, rubbish bins etc. Using our Script-UI you can further expand the amount of different station objects and station types. Once you press on the Unity play button, the algorithm starts to generate the station environments by randomly generating and placing the human models and station objects along the subway station. Once the scenario is generated, the algorithm takes a screenshot and saves the image to a predefined folder within the project folder. Since we are training a semantic segmentation algorithm we also need to generate the ground truth labels. To do so, our algorithm replaces all station objects and the station itself with white textured versions of those objects. The human models are replaced with green, yellow and red versions of the human models based on their location within the subway station. If they are staying on the railroads, they are painted in red. If they are staying in front of the security line, they are painted in yellow. In all other cases, the human models are replaced with green ones. Finally, our algorithm takes another screenshot and stores the image as the ground truth label in a separate folder within the project folder.


<p align="center">                                                                                                                    
   <img align="left" width="390" height="200" src="Result%20images/Simulation%20images/Human%20models.png">
<img align="right" width="390" height="200" src="Result%20images/Simulation%20images/Station%20objects.png">
</p>


<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>

<p align="right">                                                                                                                    
   <img align="right" width="390" height="" src="Result%20images/Script-UI%20Screenshot.png">
</p>
On the right side you can see our Script-UI which is used to controll the simulation. It contains a camera object which is used to take the screenshots. Upon activating the first checkbox "Use Distribution" the simulation produces a more realistic scenario where the persons are distributed along the station using a real data-based distribution of persons (you can read more about it in the report). The second checkbox "Create Samples" is used to create a random scene from the simulation and freezes it (this is mostly used for testing purposes once we add new objects to the simulation). 
Because of memory space issues we had to implement our algorithm to work only at one station type (out of 50) at the time. You can specify which station type should be used as background in the last option called "Type Index". 
The following 8 options (starting with "Min Persons" and ending with "Max Snacks") are used to threshold the algorithm to how many instances of the different object classes should be generated. The min options specify the minimum number and the max options specify the maximum number of objects which are generated for the scenario. The algorithm picks randomly a number in between. 
The following dropdown options are simple lists which are used to store each gameobject which will be used for generating the scene. It is very important that every gameobject is assigned to the correct list. Notice that there are 4 different "Chars" lists. This is because every human model has to contain also it green, yellow and red painted twin in the segmentation scenario. This also applies for other objects from the station.


<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>

### Datasets ####

* The full dataset with uniform distributed persons on the stations is available [here](https://drive.google.com/drive/folders/1QWc8qHPNCtirC2PKcNBOb_mIcEepQ4jy?usp=sharing)
* The full dataset with our own distribution (more information in our) for persons on the stations is available [here](https://drive.google.com/drive/folders/1JZ6PK5veVjP6tqiLZnq6TZdRxIqgvGTe?usp=sharing)  
### Semantic Segmentation using [SegNet](https://arxiv.org/pdf/1505.07293.pdf) ###

For detecting a hazard on the subway station we are using semantic segmentation to classify each pixel of an image to one of the following classes:

* white - background
* black - security line
* green - character in save area
* yellow - character near the dangerous area
* red - character in the dangerous area

**Training**

To train the [SegNet](https://arxiv.org/pdf/1505.07293.pdf) there are 2 scripts available:

* ```SUBWAY_SEGMENTATION.py```
* ```Subwaystation_Segmentation.ipynb```

For both you only need a folder with the input images and another one with the target images.
To start the training you only should execute:

    python3 SUBWAY_SEGMENTATION.py
    
with the following parameters

* ```--input path``` to img input data directory
* ```--target path``` to img target data directory
* ```--content path``` where the train/validation tensors, model_weights, losses, validation will be saved, ```default="/"```
* ```--train_tensor_size``` number of images per training_tensor (should be True: ```train_tensor_size % batch_size == 0```)
* ```--val_tensor_size``` help='number of images per training_tensor (should be True: ```train_tensor_size % batch_size == 0```)
* ```--num_train_tensors``` help='number of train tensors (should be True: ```train_tensor_size * num_train_tensors + val_tensor_size == |images|```)
* ```--model_weights``` path where your model weights will be loaded, if not defined new weights will initialized
* ```--epochs number``` of training epochs, ```default=50```
* ```--batch_size``` batch size for training, ```default=8```
* ```--learn_rate``` learning rate for training, ```default=0.0001```
* ```--momentum``` momentum for stochastic gradient descent, ```default=0.9```
* ```--save_cycle``` save model, loss, validation every save_cycle epochs, ```default=5```
* ```--weight_decay``` weight_decay for stochastic gradient descent, ```default= 4e5```

example execution with 41.000 input/target images:

    python3 test_parse.py --input=data/training --target=data/target --content=data/output --train_tensor_size=2000 --val_tensor_size=1000 --num_train_tensors=20  model_weights=model.pt
    
configuration from example execution:

* ```input_path=data/training```
* ```target_path=data/target```
* ```content_path=data/output```
* ```batch_size=2000```
* ```train_tensor_size=20```
* ```val_tensor_size=1000```
* ```num_train_tensors=20```
* ```model_weights=model.pt```
* ```load_model=True```
* ```learn_rate=0.0001```
* ```momentum=0.9```
* ```weight_decay=400000.0```
* ```total_epochs=50```
* ```save_cycle=5```

for ```Subwaystation_Segmentation.ipynb``` its the equivalent for google-colab. You can set all the parameters in the configuration cell.

<a href="https://colab.research.google.com/github/alen-smajic/Subway-Station-Hazard-Detection/blob/main/Colab_Notebooks/Subwaystation_Segmentation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Predict**

To predict from the trained model we provide the google-colab notebook ```Subway_Segmentation_Predict.ipynb```, which is self explained.

You only have to change the following paths:

* model weights
* input image
* target image

check it out 

<a href="https://colab.research.google.com/github/alen-smajic/Subway-Station-Hazard-Detection/blob/main/Colab_Notebooks/Subway_Segmentation_Predict.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Results ##
<img align="center" width="1000" height="" src="Result%20images/Segmentation%20images/Segmentation%205.jpg">
<img align="center" width="1000" height="" src="Result%20images/Segmentation%20images/Segmentation%204.png">

<img align="center" width="1000" height="" src="Result%20images/Simulation%20images/Station%20and%20ground%20truth%202.jpg">

<img align="center" width="1000" height="" src="Result%20images/Simulation%20images/Station%20and%20ground%20truth%201.jpg">
