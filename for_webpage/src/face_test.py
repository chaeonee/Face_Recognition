"""Performs face alignment and calculates L2 distance between the embeddings of images.
    https://github.com/davidsandberg/facenet """

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


from PIL import Image
import face_recognition

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import facenet
import align.detect_face

# img_name = '/Users/onee/Desktop/face_test/kie_dd.jpg'
def find_face(img_name):
    model = '../20180402-114759/'
    image_path = 'data/'
    image_size = 160
    margin = 44
    gpu_memory_fraction = 1.0
    
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(img_name)
#    image = face_recognition.load_image_file('/Users/onee/Desktop/face_test/ChildCarev2-20181209031241.jpg')
    
    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)
    
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    
    face_images = []
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
#        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_images.append(face_image)
#        pil_image = Image.fromarray(face_image)
#        pil_image.show()  
        
        
    person_names = compare(face_images, model, image_path, image_size, margin, gpu_memory_fraction)
    
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