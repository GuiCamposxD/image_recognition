#!/bin/bash

# Check if an image filename is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <image_filename>"
  exit 1
fi

# Set the image file and API URL
IMAGE_FILE="$1"
API_URL="http://15.229.5.21:5000/predict"

# Check if the file exists
if [ ! -f "$IMAGE_FILE" ]; then
  echo "File not found: $IMAGE_FILE"
  exit 1
fi

# Send the image file to the API using curl
RESPONSE=$(curl -m 5 -X POST -F "file=@$IMAGE_FILE" "$API_URL")

# Print the API response
echo "API Response: $RESPONSE"

# curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:5000/predict
# curl -X POST -F "file=@/home/ubuntu/flaskapp/testing/images/banana.jpg" http://98.83.161.98:5000/predict