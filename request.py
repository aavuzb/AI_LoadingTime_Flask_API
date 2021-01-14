import base64
import cv2
import json
import requests
import io
import argparse
import json
from flask import jsonify


path = 'loading_2.jpg'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True,
	help="name of the user")
args = vars(ap.parse_args())

response = requests.post('http://localhost:3000/', data={'data': args["image_path"]})

img = cv2.imread(path, 1)
height, width, _ = img.shape

print("height: ", height)
print("width: ", width)