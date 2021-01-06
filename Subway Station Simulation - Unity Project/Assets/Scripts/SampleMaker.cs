using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using UnityEditor;
using UnityEngine;

public class SampleMaker : MonoBehaviour
{
    public Camera myCamera;
    public GameObject cameraObject;

    public bool useDistribution;
    public bool createSamples;
    public int numberOfStations;
    public int numberOfScenes;

    public int numberOfTypes;

    public int minPersons;
    public int maxPersons;

    private float saveY = 5.5f; //11Wand, 5Schiene 
    private float dangerY = 2.0f;

    public int minChairs = 2;
    public int maxChairs = 5;

    public int minBins = 1;
    public int maxBins = 3;

    public int minSnacks = 0;
    public int maxSnacks = 2;

    private int distance = 1;
    private int distanceObj = 30;

    //public GameObject trainstation;
    //public GameObject trainstation_White;

    public List<GameObject> stationTypes;
    public List<GameObject> stationTargetTypes;

    public List<GameObject> chars; //normal characters
    public List<GameObject> charsD; //characters in dangerous area
    public List<GameObject> charsS; //characters in safe area
    public List<GameObject> charsN; //characters near dangerous area

    public List<GameObject> dogs; //normal dogs
    public List<GameObject> dogsD; //dogs in dangerous area
    public List<GameObject> dogsS; //dogs in safe area
    public List<GameObject> dogsN; //dogs near dangerous area

    public List<GameObject> chairsWall;
    public List<GameObject> chairsWallTarget;
    public List<GameObject> chairsCenter;
    public List<GameObject> chairsCenterTarget;

    public List<GameObject> binsWall;
    public List<GameObject> binsWallTarget;
    public List<GameObject> binsCenter;
    public List<GameObject> binsCenterTarget;

    public List<GameObject> snackMachinesWall;
    public List<GameObject> snackMachinesWallTarget;
    public List<GameObject> snackMachinesCenter;
    public List<GameObject> snackMachinesCenterTarget;

    public List<GameObject> stairsWall;
    public List<GameObject> stairsWallTarget;
    public List<GameObject> stairsCenter;
    public List<GameObject> stairsCenterTarget;

    public GameObject elevator;
    public GameObject elevatorTarget;

    public GameObject entrance;
    public GameObject entranceTarget;

    private List<StationChar> stationChars = new List<StationChar>();
    private List<Station> stations = new List<Station>();
    
    private GameObject currentObjs;
    private GameObject currentObjsTarget;
    private GameObject currentChars;

    private List<int> personChoices = new List<int>();

    //private GameObject currentCharsTarget;

    //private float normalZ = 5.5f;
    //private float dangerZ = 1.8f;


    //private bool isScene1 = true;

    private int objNumberStart;
    private int objNumber;
    private int typeNumber;

    private int startX = -63;
    private int stopX = 215;
    
    //private int startZ;
    //private int stopZ;

    private GameObject trainstation;
    private GameObject trainstationTarget;

    //40 pause fahrstuhl 56 X-Achse

    private int securityLineZ = 2; //(danger ab 3 unten Spawn ab 4 oben spawnen);

    Station station;

    private bool isReady;
    public int typeIndex;  //this  should be public
    private int typeEnd;
    private int stationIndex;
    private int sceneIndex;

    // Start is called before the first frame update

    void Start()
    {
        for(int i=0; i < 81; i++)
        {
            personChoices.Add(i);
        }
        personChoices.Add(100);
        personChoices.Add(200);

        
        
        //Random.seed = 42;
        //scene = new GameObject("Scene");
        currentObjs = new GameObject("Objects");
        currentObjsTarget = new GameObject("TargetObjects");
        currentChars = new GameObject("Chars");
        trainstation = new GameObject("Trainstation");
        trainstationTarget = new GameObject("TrainstationTarget");
        //currentCharsTarget = new GameObject("TargetChars");

        stations.Add(new Station(-63, 215, -28, 12, -65, 30, 0, 30, 90, 0, 0, true, false, 0, true, -3.5f, 3.5f, 0 ,0, 0, 0, 1));
        stations.Add(new Station(-63, 215, -42, 19, -65, 30, 0, 30, 90, 0, 0, true, false, 0, true, -3.5f, 3.5f, 0, 0, 0, 0, 1));
        stations.Add(new Station(-63, 215, -47, 26, -65, 30, 0, 30, 90, 0, 0, true, false, 0, true, -3.5f, 3.5f, 0, 0, 0, 0, 1));
        stations.Add(new Station(-63, 215, -85, 26, -65, 30, -60, 30, 90, 0, -30, false, false, 0, true, -53.5f, -60.5f, -3.5f, 3.5f, 0, 0, 2));
        stations.Add(new Station(-63, 215, -66, 19, -65, 30, -50, 30, 90, 0, -25, false, false, 0, true, -46.5f, -53.5f, -3.5f, 3.5f, 0, 0, 2));

        stations.Add(new Station(-63, 215, -47, 75, -65, 30, 0, 30, 90, 0, 0, true, true, 79, true, -3.5f, 3.5f, 33, 26, 0, 0, 3));
        stations.Add(new Station(-63, 215, -68, 80, -65, 30, -50, 30, 90, 0, -23, false, true, 53.5f, true, -44, -51, -3.5f, 3.5f, 26, 33, 4));
        stations.Add(new Station(-63, 215, -85, 26, -65, 30, 0, 30, 90, 0, -14, false, true, -41, false, -53.5f, -60.5f, -3.5f, 3.5f, 0, 0, 2));
        stations.Add(new Station(-63, 215, -70, 19, -65, 30, -50, 30, 90, 0, -24, false, false, 0, false, -46.5f, -53.5f, -3.5f, 3.5f, 0, 0, 2));
        stations.Add(new Station(-63, 215, -68, 24, -65, 30, -50, 30, 90, 0, -33, false, true, -15, false, -44, -51, -3.5f, 3.5f,0, 0, 2));

        /*
        stations.Add(new Station(-63, 215, -28, 12, -65, 20, 0, 30, 90, 0, 0, true, false, 0, true, -3.5f, 3.5f, 0, 0, 0, 0, 1));
        stations.Add(new Station(-63, 215, -42, 19, -65, 20, 0, 30, 90, 0, 0, true, false, 0, true, -3.5f, 3.5f, 0, 0, 0, 0, 1));
        stations.Add(new Station(-63, 215, -47, 26, -65, 20, 0, 30, 90, 0, 0, true, false, 0, true, -3.5f, 3.5f, 0, 0, 0, 0, 1));
        stations.Add(new Station(-63, 215, -85, 26, -65, 30, -30, 30, 90, 0, -30, false, false, 0, true, -53.5f, -60.5f, -3.5f, 3.5f, 0, 0, 2));
        stations.Add(new Station(-63, 215, -66, 19, -65, 30, -25, 30, 90, 0, -25, false, false, 0, true, -46.5f, -53.5f, -3.5f, 3.5f, 0, 0, 2));

        stations.Add(new Station(-63, 215, -47, 75, -65, 30, 0, 30, 90, 0, 0, true, true, 79, true, -3.5f, 3.5f, 33, 26, 0, 0, 3));
        stations.Add(new Station(-63, 215, -68, 80, -65, 30, -24, 30, 90, 0, -23, false, true, 53.5f, true, -44, -51, -3.5f, 3.5f, 26, 33, 4));
        stations.Add(new Station(-63, 215, -85, 26, -65, 30, 0, 30, 90, 0, -14, false, true, -41, false, -53.5f, -60.5f, -3.5f, 3.5f, 0, 0, 2));
        stations.Add(new Station(-63, 215, -70, 19, -65, 30, -24, 30, 90, 0, -24, false, false, 0, false, -46.5f, -53.5f, -3.5f, 3.5f, 0, 0, 2));
        stations.Add(new Station(-63, 215, -68, 24, -65, 30, -50, 30, 90, 0, -33, false, true, -15, false, -44, -51, -3.5f, 3.5f, 0, 0, 2));
        */

        if (createSamples)
        {
            typeEnd = typeIndex + 2;
            isReady = true;
            stationIndex = 0;
            sceneIndex = 0;
            //MakeSamples();
        }
        else
        {
            int stationNum = Random.Range(0, stationTypes.Count);
            SpawnStationType(stationNum);
            CreateStation();
            CreateScene(50);
        }
        

        //MakeSamples();
        /*
        SpawnObject(5, 0, 5.5f, 0, true);
        SpawnObject(8, -3, 5.5f, 0, true);
        ResetScene();
        SpawnObject(8, -3, 5.5f, 0, true);
        SpawnObject(8, -3, 5.5f, 0, true);
        */

    }

    // Update is called once per frame
    void Update()
    {
        
        if (createSamples)
        {
            StartCoroutine("MakeSamples");
        }
        
    }

    private void SpawnStationType(int stationNum)
    {
        GameObject curStation = Instantiate(stationTypes[stationNum]) as GameObject;
        curStation.transform.position = new Vector3(0, 0, 0);
        curStation.transform.parent = trainstation.transform;

        int stationIndex = stationNum / numberOfTypes;

        GameObject curStationTarget = Instantiate(stationTargetTypes[stationNum]) as GameObject;
        curStationTarget.transform.position = new Vector3(0, 0, 0);
        curStationTarget.transform.parent = trainstationTarget.transform;

        station = stations[stationIndex];
    }

    private void CreateStation()
    {
        trainstationTarget.gameObject.SetActive(false);
        currentObjsTarget.gameObject.SetActive(false);

        cameraObject.transform.position = new Vector3(station.camPosX, station.camPosY, station.camPosZ);
        cameraObject.transform.eulerAngles = new Vector3(station.camRotX, station.camRotY, station.camRotZ);

        int chairNum = Random.Range(minChairs, maxChairs + 1);
        int binNum = Random.Range(minBins, maxBins + 1);
        int snackNum = Random.Range(minSnacks, maxSnacks + 1);
        int wallType = station.hasBigs ? Random.Range(0, 3) : -1;


        if (station.oneRail)
        {

            switch (wallType) //spawn entrance, or stairs
            {
                case 0:
                    GameObject curEntrance = Instantiate(entrance) as GameObject;
                    curEntrance.transform.position = new Vector3(stopX - 40, saveY, station.startZ);
                    curEntrance.transform.parent = currentObjs.transform;

                    GameObject curEntranceTarget = Instantiate(entranceTarget) as GameObject;
                    curEntranceTarget.transform.position = new Vector3(stopX - 40, saveY, station.startZ);
                    curEntranceTarget.transform.parent = currentObjsTarget.transform;
                    break;

                case 1:
                    int index = Random.Range(0, stairsCenter.Count);
                    GameObject curEscalator = Instantiate(stairsWall[index]) as GameObject;
                    curEscalator.transform.position = new Vector3(station.stopX, saveY, station.startZ);
                    curEscalator.transform.parent = currentObjs.transform;

                    GameObject curEscalatorTarget = Instantiate(stairsWallTarget[Random.Range(0, stairsCenter.Count)]) as GameObject;
                    curEscalatorTarget.transform.position = new Vector3(station.stopX, saveY, station.startZ);
                    curEscalatorTarget.transform.parent = currentObjsTarget.transform;
                    break;

                default:
                    break;
            }

            SpawnObject(saveY + 2, station.startZ - 2, chairsWall, chairsWallTarget, chairNum, false);
            SpawnObject(saveY, station.startZ, binsWall, binsWallTarget, binNum, false);
            SpawnObject(saveY - 0.7f, station.startZ, snackMachinesWall, snackMachinesWallTarget, snackNum, false);

            if (station.bothSides)
            {
                SpawnObject(saveY + 2, station.spawnPoint2 - 2, chairsWall, chairsWallTarget, chairNum, true);
                SpawnObject(saveY, station.spawnPoint2, binsWall, binsWallTarget, binNum, true);
                SpawnObject(saveY - 0.7f, station.spawnPoint2, snackMachinesWall, snackMachinesWallTarget, snackNum, true);
            }
            

        }
        else
        {
            switch (wallType) //spawn entrance, or stairs
            {
                case 0:
                    GameObject curElevator = Instantiate(elevator) as GameObject;
                    curElevator.transform.position = new Vector3(stopX - 40, saveY, station.center);
                    curElevator.transform.parent = currentObjs.transform;

                    GameObject curElevatorTarget = Instantiate(elevatorTarget) as GameObject;
                    curElevatorTarget.transform.position = new Vector3(stopX - 40, saveY, station.center);
                    curElevatorTarget.transform.parent = currentObjsTarget.transform;
                    break;

                case 1:
                    int index = Random.Range(0, stairsCenter.Count);
                    GameObject curEscalator = Instantiate(stairsCenter[index]) as GameObject;
                    curEscalator.transform.position = new Vector3(station.stopX, saveY, station.center);
                    curEscalator.transform.parent = currentObjs.transform;

                    GameObject curEscalatorTarget = Instantiate(stairsCenterTarget[index]) as GameObject;
                    curEscalatorTarget.transform.position = new Vector3(station.stopX, saveY, station.center);
                    curEscalatorTarget.transform.parent = currentObjsTarget.transform;
                    break;

                default:
                    break;
            }

            SpawnObject(saveY, station.center, chairsCenter, chairsCenterTarget, chairNum, false);
            SpawnObject(saveY, station.center, binsCenter, binsCenterTarget, binNum, false);
            SpawnObject(saveY - 0.7f, station.center, snackMachinesCenter, snackMachinesCenterTarget, snackNum, false);

            if (station.bothSides)
            {
                SpawnObject(saveY, station.spawnPoint2, chairsCenter, chairsCenterTarget, chairNum, false);
                SpawnObject(saveY, station.spawnPoint2, binsCenter, binsCenterTarget, binNum, false);
                SpawnObject(saveY - 0.7f, station.spawnPoint2, snackMachinesCenter, snackMachinesCenterTarget, snackNum, false);
            }
            
        }
        
    }

    private void SpawnObject(float y, float z, List<GameObject> objects, List<GameObject> objectsTarget,  int number, bool rotate180)
    {
        Collider[] hitColliders;
        int x;
        GameObject curObj;
        GameObject curObjTarget;
        int index;
        bool hasCollision;
        int count;

        for (int i = 0; i < number; i++)
        {
            count = 0;
        
            do
            {
                hasCollision = false;
                x = Random.Range(startX, stopX);
                
                foreach (Transform child in currentObjs.transform)
                {
                    
                    if (Mathf.Abs(child.transform.position.x - x) < distanceObj && Mathf.Abs(child.transform.position.z - z) < 5)
                    {
                        hasCollision = true;
                    }
                    if ((child.name == "EscalatorCenter(Clone)" || child.name == "EscalatorWall(Clone)" || child.name == "stairsCenter(Clone)" || child.name == "stairsWall(Clone)") && Mathf.Abs(child.transform.position.x - x) < 70)
                    {
                        hasCollision = true;
                    }
                }
                count++;
                
                //hitColliders = Physics.OverlapSphere(new Vector3(x, y, z), distanceObj);

                /*
                Debug.Log(objects[index].transform.lossyScale);
                Debug.Log(i);
                Debug.Log(hitColliders.Length);
                Debug.Log(hitColliders.Length != 0);
                if(hitColliders.Length != 0)
                {
                    for(int j = 0; j < hitColliders.Length; j++)
                    {
                        Debug.Log(hitColliders[j].name);
                    }
                }
                */

            } while (hasCollision && count < 50); //hitColliders.Length != 0);

            //Debug.Log("x" + x.ToString());
            index = Random.Range(0, objects.Count);
            curObj = Instantiate(objects[index]) as GameObject;
            curObjTarget = Instantiate(objectsTarget[index]) as GameObject;
            curObj.transform.position = new Vector3(x, y, z);
            curObjTarget.transform.position = new Vector3(x, y, z);

            if (rotate180) 
            {
                curObj.transform.eulerAngles = new Vector3(curObj.transform.eulerAngles.x, 180, curObj.transform.eulerAngles.z);
                curObjTarget.transform.eulerAngles = new Vector3(curObjTarget.transform.eulerAngles.x, 180, curObjTarget.transform.eulerAngles.z);
            }
            
            curObj.transform.parent = currentObjs.transform;
            curObjTarget.transform.parent = currentObjsTarget.transform;
        }
    }

    private void FixedUpdate()
    {
     
    }

    private void SpawnChar(int index, int x, float y, float z, int yRot, int danger)
    {
        GameObject currentObj;
        switch (danger)
        {
            case 0:
                currentObj = Instantiate(charsD[index]) as GameObject;
                break;

            case 1:
                currentObj = Instantiate(charsS[index]) as GameObject;
                break;

            case 2:
                currentObj = Instantiate(charsN[index]) as GameObject;
                break;


            default:
                currentObj = Instantiate(chars[index]) as GameObject;
                break;
        }
        
        currentObj.transform.position = new Vector3(x, y, z);
        currentObj.transform.eulerAngles = new Vector3(currentObj.transform.eulerAngles.x, yRot, currentObj.transform.eulerAngles.z);
        //Debug.Log(currentObj.transform.rotation);
        currentObj.transform.parent = currentChars.transform;
    }

    private void SpawnDog(int index, int x, float y, float z, int yRot, int danger)
    {
        GameObject currentObj;
        switch (danger)
        {
            case 0:
                currentObj = Instantiate(dogsD[index]) as GameObject;
                break;

            case 1:
                currentObj = Instantiate(dogsS[index]) as GameObject;
                break;

            case 2:
                currentObj = Instantiate(dogsN[index]) as GameObject;
                break;


            default:
                currentObj = Instantiate(dogs[index]) as GameObject;
                break;
        }

        currentObj.transform.position = new Vector3(x, y, z);
        currentObj.transform.eulerAngles = new Vector3(currentObj.transform.eulerAngles.x, yRot, currentObj.transform.eulerAngles.z);
        //Debug.Log(currentObj.transform.rotation);
        currentObj.transform.parent = currentChars.transform;
    }

    private void CreateScene(int number)
    {
        float objZ = 0;
        float objY = 0;
        int objX = 0;
        int danger = 0;
        int objIndex = 0;
        int objYRotation = 0;
        Collider[] hitColliders;
        int count;
        bool save;

        for (int i = 0; i < number; i++)
        {
            count = 0;

            do
            {
      
                objX = Random.Range(station.startX, station.stopX);
                
                if (useDistribution)
                {
                    save = Random.Range(0, 10000) > 0;
                    objZ = GaussianZ(save);
                }
                else
                {
                    objZ = Random.Range(station.startZ, station.stopZ);
                }
                

                switch (station.type)
                {
                    case 1:
                        if (objZ < station.yellowZone)
                        {
                            objY = saveY;
                            danger = 1;
                        }
                        else if (objZ < station.redZone && objZ >= station.yellowZone)
                        {
                            objY = saveY;
                            danger = 2;
                        }
                        else
                        {
                            objY = dangerY;
                            danger = 0;
                        }
                        break;

                    case 2:

                        if (objZ < station.redZone || objZ > station.redZone2)
                        {
                            objY = dangerY;
                            danger = 0;
                        }
                        else if (objZ >= station.yellowZone && objZ <= station.yellowZone2)
                        {
                            objY = saveY;
                            danger = 1;
                        }
                        else
                        {
                            objY = saveY;
                            danger = 2;
                        }
                        break;

                    case 3:

                        if ((objZ >= station.yellowZone && objZ <= station.redZone) || (objZ <= station.yellowZone2 && objZ >= station.redZone2))
                        {
                            objY = saveY;
                            danger = 2;
                        }
                        else if (objZ > station.redZone && objZ < station.redZone2)
                        {
                            objY = dangerY;
                            danger = 0;
                        }
                        else
                        {
                            objY = saveY;
                            danger = 1;
                        }
                        break;

                    case 4:

                        if (objZ < station.redZone || (objZ > station.redZone2 && objZ < station.redZoneCB))
                        {
                            objY = dangerY;
                            danger = 0;
                        }
                        else if ((objZ >= station.redZone && objZ <= station.yellowZone) || (objZ >= station.yellowZone2 && objZ <= station.redZone2) || (objZ >= station.redZoneCB && objZ <= station.yellowZoneCB))
                        {
                            objY = saveY;
                            danger = 2;
                        }
                        else
                        {
                            objY = saveY;
                            danger = 1;
                        }
                        break;

                    default:
                        objY = saveY;
                        danger = 1;
                        break;
                }
                count++;
                //objY = (objZ < 3) ? saveY : dangerY;
                hitColliders = Physics.OverlapSphere(new Vector3(objX, objY, objZ), distance);
            } while ((hitColliders.Length != 0) && count < 50);
    

            objIndex = Random.Range(0, chars.Count);
            objYRotation = Random.Range(0, 360);

            SpawnChar(objIndex, objX, objY, objZ, objYRotation, -1);

            stationChars.Add(new StationChar(objIndex, objX, objY, objZ, objYRotation, danger));

            if ((useDistribution && Random.Range(0,200) == 0))
            {
                objZ = GaussianZ(false);
                objX = Random.Range(station.startX, station.stopX);
                objIndex = Random.Range(0, chars.Count);
                objYRotation = Random.Range(0, 360);
                danger = 0;
                SpawnChar(objIndex, objX, dangerY, objZ, objYRotation, -1);
                stationChars.Add(new StationChar(objIndex, objX, dangerY, objZ, objYRotation, danger));

            }


            
        }
        /*
        if(useDistribution)
        {
            if (Random.Range(0, 50) == 0)
            {
                objIndex = Random.Range(0, dogs.Count);
                SpawnDog(objIndex, objX + 2, objY, objZ, objYRotation, -1);
            }
        }
        else
        {
            if (Random.Range(0, 2) == 0)
            {
                objIndex = Random.Range(0, dogs.Count);
                SpawnDog(objIndex, objX + 2, objY, objZ, objYRotation, -1);
            }
        }
        */
    }

    private float GaussianZ(bool save)
    {
        bool isOne = Random.Range(0, 200) == 0;
        float objZ = 0;

        switch (station.type)
        {
            case 1:
                if(save)
                {
                    objZ = NextGaussian(CalcMean(station.startZ, station.redZone + 1), CalcVariance(station.startZ, station.redZone+1), station.startZ, station.redZone+1);
                }
                else
                {
                    objZ = Random.Range(station.redZone + 1, station.stopZ);
                }
                break;

            case 2:
                if (save)
                {
                    objZ = NextGaussian(CalcMean(station.redZone, station.redZone2 + 1), CalcVariance(station.redZone, station.redZone2+1), station.redZone, station.redZone2+1);
                }
                else
                {
                    if (isOne) 
                    {
                        objZ = Random.Range(station.startZ, station.redZone);
                    }
                    else
                    {
                        objZ = Random.Range(station.redZone2+1, station.stopZ);
                    }
                }
                break;

            case 3:
                if (save)
                {
                    if (isOne)
                    {
                        objZ = NextGaussian(CalcMean(station.startZ, station.redZone + 1), CalcVariance(station.startZ, station.redZone+1), station.startZ, station.redZone+1);
                    }
                    else
                    {
                        objZ = NextGaussian(CalcMean(station.redZone2 + 1, station.stopZ), CalcVariance(station.redZone2+1, station.stopZ), station.redZone2+1, station.stopZ);
                    }
                    
                }
                else
                {
                    objZ = Random.Range(station.redZone + 1, station.redZone2);
                }
                break;
            
            case 4:
                if (save)
                {
                    if (isOne)
                    {
                        objZ = NextGaussian(CalcMean(station.redZone, station.redZone2 + 1),CalcVariance(station.redZone, station.redZone2 + 1), station.redZone, station.redZone2 + 1);
                    }
                    else
                    {
                        objZ = NextGaussian(CalcMean(station.redZoneCB, station.stopZ), CalcVariance(station.redZoneCB, station.stopZ), station.redZoneCB, station.stopZ);
                    }
                }
                else
                {
                    if (isOne)
                    {
                        objZ = Random.Range(station.startX, station.redZone);
                    }
                    else
                    {
                        objZ = Random.Range(station.redZone2+1, station.redZoneCB);
                    }
                }
                break;

            default:
                break;
        }
        return objZ;
    }

    private float CalcVariance(float min, float max)
    {
        return Mathf.Abs(min - max) / 4;
    }

    private float CalcMean(float min, float max)
    {
        return (min + max) / 2;
    }

    IEnumerator MakeSamples()
    {
        if (isReady && (typeIndex < typeEnd))
        {
            Debug.Log("Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString());
            isReady = false;

            if(stationIndex == 0 && sceneIndex == 0)
            {
                ResetStation();
                SpawnStationType(typeIndex);
                typeIndex++;

            }
            if (sceneIndex == 0){

                ResetStationObjects();
                CreateStation();
                stationIndex = ((stationIndex + 1) == numberOfStations) ? 0 : stationIndex + 1;
            }

            int numberOfPersons;

            // numberOfPersons = Random.Range(minPersons, maxPersons);
            int personIndex = Random.Range(0, personChoices.Count);
            numberOfPersons = personChoices[personIndex];

            myCamera.transform.position = new Vector3(station.startX, myCamera.transform.position.y, myCamera.transform.position.z);
            myCamera.transform.eulerAngles = new Vector3(station.camRotX, station.camRotY, station.camRotZ);

            RenderSettings.ambientIntensity = 2.5f;
            CreateScene(numberOfPersons);
            trainstation.active = true;
            currentObjs.active = true;
            trainstationTarget.active = false;
            currentObjsTarget.active = false;
            MakeScreenshot("/Input/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "front.jpg");
            //MakeScreenshot("D:/Input/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "front.jpg");
            myCamera.Render();
            myCamera.transform.position = new Vector3(station.stopX, myCamera.transform.position.y, myCamera.transform.position.z);
            myCamera.transform.eulerAngles = new Vector3(station.camRotX, station.camRotY * -1, station.camRotZ);

            MakeScreenshot("/Input/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "back.jpg");
            //MakeScreenshot("D:/Input/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "back.jpg");
            myCamera.Render();
            ResetChars();

            myCamera.transform.position = new Vector3(station.startX, myCamera.transform.position.y, myCamera.transform.position.z);
            myCamera.transform.eulerAngles = new Vector3(station.camRotX, station.camRotY, station.camRotZ);

            RenderSettings.ambientIntensity = 8;
            CreateTarget();
            trainstation.active = false;
            currentObjs.active = false;
            trainstationTarget.active = true;
            currentObjsTarget.active = true;
            MakeScreenshot("/Target/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "front.jpg");
            //MakeScreenshot("D:/Target/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "front.jpg");
            myCamera.Render();
            myCamera.transform.position = new Vector3(station.stopX, myCamera.transform.position.y, myCamera.transform.position.z);
            myCamera.transform.eulerAngles = new Vector3(station.camRotX, station.camRotY * -1, station.camRotZ);

            MakeScreenshot("/Target/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "back.jpg");
            //MakeScreenshot("D:/Target/Station" + typeIndex.ToString() + "Type" + stationIndex.ToString() + "Scene" + sceneIndex.ToString() + "back.jpg");
            myCamera.Render();
            ResetChars();

            sceneIndex = ((sceneIndex + 1) == numberOfScenes) ? 0 : sceneIndex +1;

            isReady = true;
        }
        yield return new WaitForSeconds(0.5f);
        
    }

    /*
    private void MakeSamples()
    {
        int numberOfPersons;

        for (int i = 0; i < stationTypes.Count; i++)
        {
            SpawnStationType(i);

            for (int j = 0; j < numberOfStations; j++)
            {
                CreateStation();

                for (int k = 0; k < numberOfScenes; k++)
                {
                    numberOfPersons = Random.Range(minPersons, maxPersons);

                    CreateScene(numberOfPersons);
                    trainstation.active = true;
                    currentObjs.active = true;
                    trainstationTarget.active = false;
                    currentObjsTarget.active = false;
                    MakeScreenshot("/Input/Station" + i.ToString() + "Type" + j.ToString() + "Scene" + k.ToString() + ".jpg");
                    myCamera.Render();
                    ResetChars();

                    CreateTarget();
                    trainstation.active = false;
                    currentObjs.active = false;
                    trainstationTarget.active = true;
                    currentObjsTarget.active = true;
                    MakeScreenshot("/Target/Station" + i.ToString() + "Type" + j.ToString() + "Scene" + k.ToString() + ".jpg");
                    myCamera.Render();
                    ResetChars();

                    //Debug.Log("ObjectNumber: " + i.ToString() + ", typeNumber: " + j.ToString());
                }

                ResetStationObjects();
            }
            ResetStation();
        }
    }
    */

    private void DeleteChilds (GameObject parentObj)
    {
        foreach (Transform child in parentObj.transform)
        {
            child.gameObject.SetActive(false);
            Destroy(child.gameObject);
        }
    }

    private void ResetStation()
    {
        DeleteChilds(trainstation);
        DeleteChilds(trainstationTarget);
    }

    private void ResetStationObjects()
    {
        foreach (Transform child in currentObjs.transform)
        {
            child.gameObject.SetActive(false);
            Destroy(child.gameObject);
        }
        foreach (Transform child in currentObjsTarget.transform)
        {
            child.gameObject.SetActive(false);
            Destroy(child.gameObject);
        }
    }

    private void MakeScreenshot(string filename)
    {
        ScreenshotHandler.TakeScreenshot_Static(1920, 1080, filename);
    }

    private void ResetChars()
    {
        foreach (Transform child in currentChars.transform)
        {
            child.gameObject.SetActive(false);
            Destroy(child.gameObject);
        }
        Update(); //Update Frame

    }

    private void CreateTarget()
    {
        foreach (StationChar obj in stationChars)
        {
            SpawnChar(obj.Index, obj.X, obj.Y, obj.Z, obj.YRot, obj.Type);
        }
        stationChars.Clear();
    }

    public static float NextGaussian()
    {
        float v1, v2, s;
        do
        {
            v1 = 2.0f * Random.Range(0f, 1f) - 1.0f;
            v2 = 2.0f * Random.Range(0f, 1f) - 1.0f;
            s = v1 * v1 + v2 * v2;
        } while (s >= 1.0f || s == 0f);

        s = Mathf.Sqrt((-2.0f * Mathf.Log(s)) / s);

        return v1 * s;
    }

    public static float NextGaussian(float mean, float standard_deviation)
    {
        return mean + NextGaussian() * standard_deviation;
    }

    public static float NextGaussian(float mean, float standard_deviation, float min, float max)
    {
        float x;
        do
        {
            x = NextGaussian(mean, standard_deviation);
        } while (x < min || x > max);
        return x;
    }
}