using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StationChar : MonoBehaviour
{
    public int Index;
    public int X;
    public float Y;
    public float Z;
    public int YRot;
    public int Type;

    public StationChar(int index, int x, float y, float z, int yRot, int type)
    {
        Index = index;
        X = x;
        Y = y;
        Z = z;
        YRot = yRot;
        Type = type;
    }
}
