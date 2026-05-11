# config.py
from dotenv import load_dotenv
import boto3
import os

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

def get_session():
    return boto3.Session(
        profile_name=AWS_PROFILE,
        region_name=AWS_REGION
    )

def get_bedrock_client():
    session = get_session()
    return session.client("bedrock-runtime")

def get_ce_client():
    session = get_session()
    return session.client("ce")