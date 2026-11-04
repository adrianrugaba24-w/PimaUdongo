import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client("s3", region_name="us-east-1",
                  config=Config(signature_version=UNSIGNED))

resp = s3.list_objects_v2(Bucket="afsis", Delimiter="/")

print("Folders:")
for p in resp.get("CommonPrefixes", []):
    print("  ", p["Prefix"])

print("Files:")
for o in resp.get("Contents", []):
    print("  ", o["Key"], o["Size"])
