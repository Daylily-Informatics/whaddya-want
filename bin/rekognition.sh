aws rekognition create-collection --collection-id companion-people
# enroll a few images (frontal, neutral light)
aws s3 cp ./faces/ s3://$BUCKET/faces/ --recursive
