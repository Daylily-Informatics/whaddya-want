"""Amazon Rekognition helper utilities."""
from __future__ import annotations

from typing import Any

import boto3


class VisionClient:
    def __init__(self, region_name: str) -> None:
        self._rekognition = boto3.client("rekognition", region_name=region_name)

    def detect_labels(self, bucket: str, key: str, max_labels: int = 10) -> list[dict[str, Any]]:
        response = self._rekognition.detect_labels(
            Image={"S3Object": {"Bucket": bucket, "Name": key}},
            MaxLabels=max_labels,
            MinConfidence=70,
        )
        return response.get("Labels", [])

    def search_faces(self, collection_id: str, bucket: str, key: str, max_faces: int = 1) -> list[dict[str, Any]]:
        response = self._rekognition.search_faces_by_image(
            CollectionId=collection_id,
            Image={"S3Object": {"Bucket": bucket, "Name": key}},
            MaxFaces=max_faces,
        )
        return response.get("FaceMatches", [])


__all__ = ["VisionClient"]
