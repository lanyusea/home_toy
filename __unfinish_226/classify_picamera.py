# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera

from PIL import Image, ImageDraw
import tensorflow


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()

  positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
  classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
  scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

  result = []

  for idx, score in enumerate(scores):
    result.append({'_id': int(classes[idx]), 'pos': positions[idx], 'score': score})

  return result

def display_result(result, frame, labels, width, height):
    draw = ImageDraw.Draw(frame)
    # position = [ymin, xmin, ymax, xmax]
    for obj in result:
        pos = obj['pos']
        _id = obj['_id']

        x1 = int(pos[1] * width)
        x2 = int(pos[3] * width)
        y1 = int(pos[0] * height)
        y2 = int(pos[2] * height)

        shape = [(x1,y1), (x2,y2)]
        draw.text((x1,y1), labels[_id], align = "left")
        draw.rectangle(shape, outline = "red")

    #cv2.imshow('Object Detection', frame)
    frame.show()
    frame.save("/home/pi/Desktop/result.jpg", "JPEG")

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = tensorflow.lite.Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
    camera.start_preview()
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        print ("spend time ", elapsed_ms, "ms with result: ")
        for result in results:
          print(result['_id'], labels[result['_id']], result['pos'], result['score'])

        display_result(results, image, labels, width, height)

    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()
