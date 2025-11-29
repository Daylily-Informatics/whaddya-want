# face_search.py
import boto3, sys
rk=boto3.client("rekognition",region_name="us-east-1"); COL="companion-people"
with open(sys.argv[1],"rb") as f:
    res=rk.search_faces_by_image(CollectionId=COL, Image={"Bytes":f.read()},
                                 FaceMatchThreshold=90, MaxFaces=1)
print(res.get("FaceMatches",[]))
