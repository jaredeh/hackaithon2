from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import uuid
import boto3
from botocore.exceptions import ClientError
import os
import yaml
import json
from stats import StorageStats

app = Flask(__name__)

def generate_fake_keys(n):
    return [{"key": str(uuid.uuid4())} for _ in range(n)]

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

aws_config = config['aws']
wasabi_config = config['wasabi']

aws_s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_config['access_key_id'],
    aws_secret_access_key=aws_config['secret_access_key'],
    region_name=aws_config['region_name']
)

# Wasabi Client
wasabi_client = boto3.client(
    's3',
    endpoint_url=wasabi_config['endpoint_url'],
    aws_access_key_id=wasabi_config['access_key_id'],
    aws_secret_access_key=wasabi_config['secret_access_key']
)

# Stats collector
stats = StorageStats()

def download_from_s3(bucket_name, object_name, file_path):
    try:
        aws_s3_client.download_file(bucket_name, object_name, file_path)
        print(f"Downloaded {object_name} from {bucket_name} to {file_path}")
    except ClientError as e:
        print(f"Failed to download {object_name} from {bucket_name}: {e}")
        raise e
    

def upload_to_wasabi(bucket_name, object_name, file_path):
    try:
        wasabi_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded {object_name} to {bucket_name} from {file_path}")
    except ClientError as e:
        print(f"Failed to upload {object_name} to {bucket_name}: {e}")
        raise e

def delete_from_s3(bucket_name, object_name):
    try:
        aws_s3_client.delete_object(Bucket=bucket_name, Key=object_name)
        print(f"Deleted {object_name} from {bucket_name}")
    except ClientError as e:
        print(f"Failed to delete {object_name} from {bucket_name}: {e}")
        raise e

def move_object_s3_to_wasabi(aws_bucket_name, aws_object_name, wasabi_bucket_name, file_path, wasabi_object_name=None):
    if wasabi_object_name is None:
        wasabi_object_name = aws_object_name

    # Download from AWS S3
    download_from_s3(aws_bucket_name, aws_object_name, file_path)
    
    # Upload to Wasabi
    upload_to_wasabi(wasabi_bucket_name, wasabi_object_name, file_path)
    
    # Optionally delete from AWS S3
    delete_from_s3(aws_bucket_name, aws_object_name)


def download_from_wasabi(bucket_name, object_name, file_path):
    try:
        wasabi_client.download_file(bucket_name, object_name, file_path)
        print(f"Downloaded {object_name} from {bucket_name} to {file_path}")
    except ClientError as e:
        print(f"Failed to download {object_name} from {bucket_name}: {e}")
        raise e

def upload_to_s3(bucket_name, object_name, file_path):
    try:
        aws_s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded {object_name} to {bucket_name} from {file_path}")
    except ClientError as e:
        print(f"Failed to upload {object_name} to {bucket_name}: {e}")
        raise e

def delete_from_wasabi(bucket_name, object_name):
    try:
        wasabi_client.delete_object(Bucket=bucket_name, Key=object_name)
        print(f"Deleted {object_name} from {bucket_name}")
    except ClientError as e:
        print(f"Failed to delete {object_name} from {bucket_name}: {e}")
        raise e

def move_object_wasabi_to_s3(wasabi_bucket_name, wasabi_object_name, aws_bucket_name, file_path, aws_object_name=None):
    if aws_object_name is None:
        aws_object_name = wasabi_object_name

    # Download from Wasabi
    download_from_wasabi(wasabi_bucket_name, wasabi_object_name, file_path)
    
    # Upload to AWS S3
    upload_to_s3(aws_bucket_name, aws_object_name, file_path)
    
    # Optionally delete from Wasabi
    delete_from_wasabi(wasabi_bucket_name, wasabi_object_name)

def list_aws_s3_objects(bucket_name):

    # List all objects in the specified bucket
    objects = aws_s3_client.list_objects_v2(Bucket=bucket_name)
    
    # Prepare the result list
    result = []

    # Extract object keys and append to the result list
    if 'Contents' in objects:
        for obj in objects['Contents']:
            result.append({
                'key': obj['Key'],
                'platform': 0  # 0 for AWS
            })
    
    # Convert the result list to a JSON string
    return json.dumps(result)

def list_wasabi_s3_objects(bucket_name):

    # List all objects in the specified bucket
    objects = wasabi_client.list_objects_v2(Bucket=bucket_name)
    
    # Prepare the result list
    result = []

    # Extract object keys and append to the result list
    if 'Contents' in objects:
        for obj in objects['Contents']:
            result.append({
                'key': obj['Key'],
                'platform': 1  # 1 for Wasabi
            })
    
    # Convert the result list to a JSON string
    return json.dumps(result)

@app.route('/list', methods=['POST'])
def handle_services():
    # Parse the incoming JSON data
    data = request.get_json()

    # Extract the filter_by_migrations field if present
    filter_by_migrations = data.get('filter_by_migrations', None)

    if filter_by_migrations:
        # Validate the filter_by_migrations input
        if not isinstance(filter_by_migrations, dict):
            return jsonify({"error": "Invalid 'filter_by_migrations' field"}), 400

        service_name = filter_by_migrations.get('service', None)
        timestamp = filter_by_migrations.get('timestamp', None)

        if not service_name or not timestamp:
            return jsonify({"error": "'filter_by_migrations' must contain 'service' and 'timestamp'"}), 400

        try:
            # Convert the timestamp to a datetime object
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return jsonify({"error": "Invalid timestamp format"}), 400

        # TODO: Query the access log table to find keys that match the service name and are within an hour of the timestamp
        # For now, return hardcoded keys
        matching_keys = generate_fake_keys(5)
        return jsonify(matching_keys), 200

    else:
        return jsonify([{"key": "lego-batman.jpeg", "platform": 0}, 
        {"key": "lego-superman.jpeg", "platform": 0}]), 200
    
    
@app.route('/migrate', methods=['POST'])
def migrate_object():
    # Parse the incoming JSON data
    data = request.get_json()

    # Extract the migration fields
    key = data.get('key')
    platform = data.get('platform')

    # hack for derek who doesn't know where it is
    if platform == 2:
        if stats.platform(key) == 0:
            platform = 1
        else:
            platform = 0

    try:
        if platform == 1:
            # Move object from AWS S3 to Wasabi
            move_object_s3_to_wasabi("hackaithon", key, "hackaithon-wasabi", f'/tmp/{key}') 
        elif platform == 0:
            move_object_wasabi_to_s3("hackaithon-wasabi", key, "hackaithon", f'/tmp/{key}')
        migration_status = "success"
    except ClientError as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    stats.log_migration(key)
    # Return the status of the migration
    return jsonify({"status": migration_status}), 200

@app.route('/get', methods=['POST'])
def get_object():
    data = request.get_json()
    key = data.get('key')
    service = data.get('service')
    
    # look up where object is based on key and service, this will give us the bucket and platform

    platform = stats.platform(key)
    bucket = 'hackaithon'
    file_path = f'./apps/{service}/static/{key}'

    try:
        if platform == 0:
            download_from_s3(bucket, key, file_path)
        elif platform == 1:
            download_from_wasabi(bucket, key, file_path)
    except ClientError as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
    stats.log_get(key,service)

    print(f"Returning file path: {file_path}")

    return jsonify({"file_path": file_path}), 200

if __name__ == '__main__':
    upload_to_s3("hackaithon", "lego-batman.jpeg", "./apps/batman/templates/lego-batman.jpeg")
    upload_to_s3("hackaithon", "lego-superman.jpeg", "./apps/superman/templates/lego-superman.jpeg")
    delete_from_wasabi("hackaithon-wasabi", "lego-batman.jpeg")
    delete_from_wasabi("hackaithon-wasabi", "lego-superman.jpeg")
    app.run(debug=True)
