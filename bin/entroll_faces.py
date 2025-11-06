# enroll_faces.py
import boto3, os
rk = boto3.client("rekognition", region_name="us-east-1")
COL="companion-people"; BUCKET=os.environ["BUCKET"]
for key in ["faces/major1.jpg","faces/major2.jpg"]:
    rk.index_faces(CollectionId=COL,
                   Image={"S3Object":{"Bucket":BUCKET,"Name":key}},
                   ExternalImageId="major", DetectionAttributes=["DEFAULT"])
print(rk.list_faces(CollectionId=COL)["Faces"])
