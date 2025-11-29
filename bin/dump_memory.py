#!/usr/bin/env python3
"""Dump short- and long-term memory DynamoDB tables as JSON."""

import argparse
import json
import os
import sys

import boto3
from boto3.dynamodb.types import TypeDeserializer


def _scan_all(client, table_name: str):
    """Read the entire table via Scan with pagination."""
    items = []
    start_key = None
    while True:
        params = {"TableName": table_name}
        if start_key:
            params["ExclusiveStartKey"] = start_key
        resp = client.scan(**params)
        items.extend(resp.get("Items", []))
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break
    return items


def _deserialize_items(raw_items):
    deserializer = TypeDeserializer()
    return [{k: deserializer.deserialize(v) for k, v in item.items()} for item in raw_items]


def _dump_table(client, table_name: str, label: str):
    raw_items = _scan_all(client, table_name)
    payload = {
        "table": table_name,
        "count": len(raw_items),
        "items": _deserialize_items(raw_items),
    }
    print(f"# {label}: {table_name}")
    print(json.dumps(payload, indent=2, default=str))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region for DynamoDB operations (default: %(default)s)",
    )
    parser.add_argument(
        "--conversation-table",
        default=os.environ.get("CONVERSATION_TABLE", ""),
        help="DynamoDB table name for short-term session memory",
    )
    parser.add_argument(
        "--ais-table",
        default=os.environ.get("AIS_MEMORY_TABLE", ""),
        help="DynamoDB table name for long-term AIS memory",
    )
    parser.add_argument(
        "--target",
        choices=["short", "long", "both"],
        default="both",
        help="Select which memory tables to dump (default: both)",
    )
    args = parser.parse_args()

    client = boto3.client("dynamodb", region_name=args.region)

    if args.target in {"short", "both"}:
        if not args.conversation_table:
            sys.exit("Missing conversation table name (set CONVERSATION_TABLE or --conversation-table)")
        _dump_table(client, args.conversation_table, "Short-term conversation memory")

    if args.target in {"long", "both"}:
        if not args.ais_table:
            sys.exit("Missing AIS memory table name (set AIS_MEMORY_TABLE or --ais-table)")
        _dump_table(client, args.ais_table, "Long-term AIS memory")


if __name__ == "__main__":
    main()
