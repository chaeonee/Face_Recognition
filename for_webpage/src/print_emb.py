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
import argparse
import facenet
import align.detect_face

def find_face():
    model = '../20180402-114759/'
    image_path = './facenet/images/'
    image_size = 160
    margin = 44
    gpu_memory_fraction = 1.0
    
    # Load the jpg file into a numpy array
#    image = face_recognition.load_image_file('/Users/onee/Desktop/face_test/FR Test Image/test/ChildCarev2-20181708103832.jpg')
#    image = face_recognition.load_image_file('/Users/onee/Desktop/face_test/ChildCarev2-20181209031241.jpg')
    
    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
#    face_locations = face_recognition.face_locations(image)
    
#    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    
#    face_images = []
#    for face_location in face_locations:
        # Print the location of each face in this image
#        top, right, bottom, left = face_location
#        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        
        # You can access the actual face itself like this:
#        face_image = image[top:bottom, left:right]
#        face_images.append(face_image)
#        pil_image = Image.fromarray(face_image)
#        pil_image.show()  
        
    face_images = []    
    compare(face_images, model, image_path, image_size, margin, gpu_memory_fraction)
        

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
            for i in range(len(emb)):
                print(image_files[i])
                print("===================================")
                print(emb[i])
                print("===================================")
            
#            nrof_images = len(image_files)
            
#            for i in range(len(input_images)):
#                dist_list = []
#                for j in range(len(input_images), len(emb)):
#                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
#                    dist_list.append(dist)
#                dist_index = dist_list.index(min(dist_list))
#                person_path = os.path.abspath(os.path.join(image_files[dist_index], '..'))
#                person_name = person_path.split('/')[-1]
                
#                print("@@@@@"+person_name)
#                pil_image = Image.fromarray(input_images[i])
#                pil_image.show()  
#                pil_image.save('/Users/onee/Desktop/'+str(i)+'_'+person_name+".jpg")
            
            
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

#def parse_arguments(argv):
#    parser = argparse.ArgumentParser()
#    
#    parser.add_argument('model', type=str, 
#        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
#    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
#    parser.add_argument('--image_size', type=int,
#        help='Image size (height, width) in pixels.', default=160)
#    parser.add_argument('--margin', type=int,
#        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
#    parser.add_argument('--gpu_memory_fraction', type=float,
#        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
#    return parser.parse_args(argv)

if __name__ == '__main__':
    find_face()
#    compare(parse_arguments(sys.argv[1:]))
