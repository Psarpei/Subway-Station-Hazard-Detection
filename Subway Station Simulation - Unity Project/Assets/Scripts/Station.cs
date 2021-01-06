using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Station : MonoBehaviour
{
    public int startX;
    public int stopX;
    public int startZ;
    public int stopZ;

    public int camPosX;
    public int camPosY;
    public int camPosZ;

    public int camRotX;
    public int camRotY;
    public int camRotZ;

    public float center;

    public bool oneRail;
    public bool bothSides;

    public float spawnPoint2;

    public bool hasBigs;

    public float yellowZone;
    public float redZone;

    public float yellowZone2;
    public float redZone2;

    public float redZoneCB;
    public float yellowZoneCB;

    public int type;

    public Station(int xStart, int xStop, int zStart, int zStop, int cpx, int cpy, int cpz, int crx, int cry, int crz, float c, bool oneR, bool bothS, float spawnP2, bool hasB, float yellowZ, float redZ, float yellowZ2, float redZ2, float redZCB, float yellowZCB, int t)
    {
        startX = xStart;
        stopX = xStop;
        startZ = zStart;
        stopZ = zStop;

        camPosX = cpx;
        camPosY = cpy;
        camPosZ = cpz;

        camRotX = crx;
        camRotY = cry;
        camRotZ = crz;

        center = c;
        oneRail = oneR;

        bothSides = bothS;
        spawnPoint2 = spawnP2;

        hasBigs = hasB;

        yellowZone = yellowZ;
        redZone = redZ;

        yellowZone2 = yellowZ2;
        redZone2 = redZ2;

        redZoneCB = redZCB;
        yellowZoneCB = yellowZCB;

        type = t;
    }
}
