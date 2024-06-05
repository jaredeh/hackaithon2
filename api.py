from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import uuid
import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

app = Flask(__name__)

def generate_fake_keys(n):
    return [{"key": str(uuid.uuid4())} for _ in range(n)]

load_dotenv()

aws_s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Wasabi Client
wasabi_client = boto3.client(
    's3',
    endpoint_url=os.getenv('WASABI_ENDPOINT_URL'),
    aws_access_key_id=os.getenv('WASABI_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('WASABI_SECRET_ACCESS_KEY')
)

def download_from_s3(bucket_name, object_name, file_path):
    aws_s3_client.download_file(bucket_name, object_name, file_path)
    

def upload_to_wasabi(bucket_name, object_name, file_path):
    wasabi_client.upload_file(file_path, bucket_name, object_name)


def delete_from_s3(bucket_name, object_name):
    aws_s3_client.delete_object(Bucket=bucket_name, Key=object_name)


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

def upload_to_s3(bucket_name, object_name, file_path):
    try:
        aws_s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded {object_name} to {bucket_name} from {file_path}")
    except ClientError as e:
        print(f"Failed to upload {object_name} to {bucket_name}: {e}")

def delete_from_wasabi(bucket_name, object_name):
    try:
        wasabi_client.delete_object(Bucket=bucket_name, Key=object_name)
        print(f"Deleted {object_name} from {bucket_name}")
    except ClientError as e:
        print(f"Failed to delete {object_name} from {bucket_name}: {e}")

def move_object_wasabi_to_s3(wasabi_bucket_name, wasabi_object_name, aws_bucket_name, file_path, aws_object_name=None):
    if aws_object_name is None:
        aws_object_name = wasabi_object_name

    # Download from Wasabi
    download_from_wasabi(wasabi_bucket_name, wasabi_object_name, file_path)
    
    # Upload to AWS S3
    upload_to_s3(aws_bucket_name, aws_object_name, file_path)
    
    # Optionally delete from Wasabi
    delete_from_wasabi(wasabi_bucket_name, wasabi_object_name)



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
        # TODO: Query the access log table to find all keys
        # For now, return hardcoded keys
        all_keys = generate_fake_keys(10)
        return jsonify(all_keys), 200
    
    
@app.route('/migrate', methods=['POST'])
def migrate_object():
    # Parse the incoming JSON data
    data = request.get_json()

    # Extract the migration fields
    key = data.get('key')
    source_bucket = data.get('source_bucket')
    destination_bucket = data.get('destination_bucket')
    platform = data.get('platform')

    # Validate the inputs
    if not key or not source_bucket or not destination_bucket:
        return jsonify({"status": "error", "message": "Missing 'key', 'source_bucket', or 'destination_bucket'"}), 400

    if platform == 0:
        # Move object from AWS S3 to Wasabi
        move_object_s3_to_wasabi(source_bucket, key, destination_bucket, f'/tmp/{key}') 
    elif platform == 1:
        move_object_wasabi_to_s3(source_bucket, key, destination_bucket, f'/tmp/{key}')
    
    migration_status = "success"

    # Return the status of the migration
    return jsonify({"status": migration_status}), 200

if __name__ == '__main__':
    app.run(debug=True)
