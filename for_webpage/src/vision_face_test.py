"""
Performs face alignment and calculates L2 distance between the embeddings of images.
    
https://github.com/davidsandberg/facenet
    
https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/vision/cloud-client/detect/detect.py
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image, ImageFont, ImageDraw
import face_recognition

from scipy import misc
import tensorflow as tf
import numpy as np
#import sys
import os
import io
import copy
import facenet
import align.detect_face

# img_name = '/Users/onee/Desktop/face_test/ChildCarev2-20181209031241.jpg'
def find_face(img_name):
    model = '../20180402-114759/'
    image_path = './data/'
    image_size = 160
    margin = 44
    gpu_memory_fraction = 1.0
    
    
    """Detects faces in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_face_detection]
    # [START vision_python_migration_image_file]
    with io.open(img_name, 'rb') as image_file:
        content = image_file.read()

    img = np.asarray(Image.open(img_name))
    image = vision.types.Image(content=content)
    # [END vision_python_migration_image_file]

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
#    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
#                       'LIKELY', 'VERY_LIKELY')
#    print('Faces:')


    face_images = []
    bb = []
    for face in faces:
#        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
#        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
#        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        
        top = face.bounding_poly.vertices[1].y
        bottom = face.bounding_poly.vertices[3].y
        left = face.bounding_poly.vertices[3].x
        right = face.bounding_poly.vertices[1].x
       
        face_image = img[top:bottom, left:right]
        face_images.append(face_image)
#        pil_image = Image.fromarray(face_image)
#        pil_image.show() 
        
        bb.append([left,top,right,bottom])
        
        
#        vertices = (['({},{})'.format(vertex.x, vertex.y)
#                    for vertex in face.bounding_poly.vertices])

#        print('face bounds: {}'.format(','.join(vertices)))
        
        
    person_names = compare(face_images, model, image_path, image_size, margin, gpu_memory_fraction)
    
    bb_color = (255,0,0)
    font = "./malgun.ttf"
    font = ImageFont.truetype(font,12)
    bb_img = Image.fromarray(img)
    draw = ImageDraw.Draw(bb_img)
    
    for i in range(len(bb)):
        draw.rectangle([(bb[i][0],bb[i][1]), (bb[i][2],bb[i][3])], outline="red")
        draw.text((bb[i][0],bb[i][1]-15), person_names[i], font=font, fill=bb_color)
        
    bb_img.save('static/result.jpg')
    
    return person_names
 

def compare(input_images, model, image_path, image_size, margin, gpu_memory_fraction):
    image_files = []
    for i in os.listdir(image_path):
        if not i == '.DS_Store':
            for f in os.listdir(os.path.join(image_path, i)):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                     image_files.append(os.path.join(image_path, i, f))
    
    images = load_and_align_data(input_images, image_files, image_size, margin, gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
#            for i in range(emb):
#                print(image_files[i])
#                print("===================================")
#                print(emb[i])
#                print("===================================")
#            
#            nrof_images = len(image_files)
            
            person_names = []
            for i in range(len(input_images)):
                dist_list = []
                for j in range(len(input_images), len(emb)):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    dist_list.append(dist)
                dist_index = dist_list.index(min(dist_list))
                person_path = os.path.abspath(os.path.join(image_files[dist_index], '..'))
                person_names.append(person_path.split('/')[-1])
#                person_name = person_path.split('/')[-1]
#                
#                print("@@@@@"+person_name)
#                pil_image = Image.fromarray(input_images[i])
#                pil_image.show()  
#                pil_image.save('/Users/onee/Desktop/'+str(i)+'_'+person_name+".jpg")
                
            return person_names
            
          
def load_and_align_data(input_images, image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    
    img_list = []
    for i in range(len(input_images)):
        aligned = misc.imresize(input_images[i], (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


if __name__ == '__main__':
    find_face()