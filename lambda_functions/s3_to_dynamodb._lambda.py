import json
import boto3
import uuid
import os
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]
    table = dynamodb.Table('heating_oil_prices')
    data = s3.get_object(Bucket=bucket, Key=key)
    contents = data['Body'].read().decode('utf-8')
    json_content = json.loads(contents)
    for item in json_content:
        table.put_item(Item={
            'id': str(uuid.uuid1()),
            'state': item['state'],
            'county': item['county'],
            'supplier': item['supplier'],
            'last_updated': item['last_updated'],
            'price150':item['price150'],
            'price300':item['price300'],
            'price500':item['price500']
        })
   
    return {
        'statusCode': 200,
        'body': json.dumps('OK')
    }