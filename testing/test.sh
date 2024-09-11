#!/bin/bash

# Check if an image filename is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <image_filename>"
  exit 1
fi

# Set the image file and API URL
IMAGE_FILE="$1"
API_URL="http://127.0.0.1:5000/predict"

# Check if the file exists
if [ ! -f "$IMAGE_FILE" ]; then
  echo "File not found: $IMAGE_FILE"
  exit 1
fi

# Send the image file to the API using curl
RESPONSE=$(curl -X POST -F "file=@$IMAGE_FILE" "$API_URL")

# Print the API response
echo "API Response: $RESPONSE"
