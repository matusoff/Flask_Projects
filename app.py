from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import json
from sklearn.cluster import KMeans
from collections import Counter


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/color-extractor', methods=['POST'])
def extract_colors():
    if 'image' not in request.files:
        return jsonify(error='No image file found'), 400

    img = Image.open(request.files['image'])
    img = img.convert('RGB')

    # convert image to numpy array
    img_array = np.array(img)

    # reshape array to 2D
    img_array = img_array.reshape(-1, img_array.shape[-1])

    # use k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(img_array)
    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)

    # sort colors by count in descending order
    sorted_indices = np.argsort(-counts)
    colors = colors[sorted_indices]
    counts = counts[sorted_indices]

    # calculate percentages
    total_pixels = img.width * img.height
    # percentages = [count / total_pixels for count in counts]
    percentages = [round(count/total_pixels*100, 2) for count in counts]


    # convert colors to hex format
    colors = ['#%02x%02x%02x' % tuple(c.astype(int)) for c in colors]

    # create response
    response = {
        'colors': colors,
        'counts': counts.tolist(),
        'percentages': percentages
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
