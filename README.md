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
Here is a description of the Script-UI.

<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>

### Semantic Segmentation using SegNet ###

Description of our Deep Learning Architecture.

## Publications ##
  * **[ResearchGate](https://www.researchgate.net/publication/344830620_Subway_Station_Hazard_Detection)**
  
## Tools ## 
* Python 3
* PyTorch Framework
* Unity
* C#
* Blender

## Results ##
<img align="center" width="1000" height="" src="Result%20images/Segmentation%20images/Segmentation%205.jpg">
<img align="center" width="1000" height="" src="Result%20images/Segmentation%20images/Segmentation%204.png">

<img align="center" width="1000" height="" src="Result%20images/Simulation%20images/Station%20and%20ground%20truth%202.jpg">

<img align="center" width="1000" height="" src="Result%20images/Simulation%20images/Station%20and%20ground%20truth%201.jpg">
