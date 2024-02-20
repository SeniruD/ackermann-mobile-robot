#!/usr/bin/env python3
# license removed for brevity

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from nltk.tokenize import word_tokenize


class TextProcessor:
    def __init__(self, vocab_file, device):
        with open(vocab_file) as f:
            self.vocab = f.read().split('", "')
        self.device = device

    def tokenize_text(self, text):
        tokens = word_tokenize(text.lower())
        return [self.vocab.index(token) for token in tokens if token in self.vocab] + [0] * (200 - len(tokens))

    def process(self, text):
        tokens = self.tokenize_text(text)
        token_tensor = torch.tensor([tokens], device=self.device)
        batch = {'rgb': [], 'depth': [], 'instruction': token_tensor}
        return batch

def rgb_image_callback(zed2_rgb_image): 
    rospy.loginfo("RGB image recieved")
    bridge = CvBridge()
    try:
        cv_rgb_image = bridge.imgmsg_to_cv2(zed2_rgb_image, "bgr8")
        # cv2.imshow("rgb image window", cv_rgb_image)
            
        (h, w, channels) = cv_rgb_image.shape
        y = (w-h)//2
        # rospy.loginfo("rgb image size: %d x %d", w, h)
        # rospy.loginfo(cv_rgb_image)
        rgb_image = cv_rgb_image[:, y:y+h]
        rgb_image = cv2.resize(rgb_image,(224,224))
        # rospy.loginfo(rgb_image.shape)
        # rgb_image_tensor = torch.tensor([rgb_image], device='cuda:0')
        # rospy.loginfo(rgb_image_tensor)
        
        batch['rgb'] = rgb_image
        save_batch(batch)
        # cv2.imshow("cropped rgb window", rgb_image)
        # cv2.waitKey(0) 
        # cv2.destropAllWindows()


    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))

    rgb_image_sub.unregister()

def depth_image_callback(zed2_depth_image): 
    rospy.loginfo("Depth image recieved")
    bridge = CvBridge()
    try:
        cv_depth_image = bridge.imgmsg_to_cv2(zed2_depth_image, "32FC1")

        # Replace inf, -inf, and NaN values with 0
        cv_depth_image[np.isinf(cv_depth_image) | np.isnan(cv_depth_image)] = 0
        # Normalize the depth image values
        # normalized_depth_image = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Normalize the depth image values
        min_val = np.min(cv_depth_image)
        max_val = np.max(cv_depth_image)

        if max_val != min_val:
            normalized_depth_image = (cv_depth_image - min_val) / (max_val - min_val)
        else:
            normalized_depth_image = np.zeros_like(cv_depth_image)
        
        # Display the normalized depth image
        # cv2.imshow("Normalized Depth Image", normalized_depth_image)
        # rospy.loginfo("depth image min: %f, max: %f", normalized_depth_image.min(), normalized_depth_image.max())
     
        # resize the depth image to 256x256
        (h, w) = normalized_depth_image.shape
        y = (w-h)//2
        depth_image = normalized_depth_image[:, y:y+h]
        depth_image = cv2.resize(depth_image,(256,256))
        # rospy.loginfo(depth_image.shape)
        # Convert NumPy array to PyTorch tensor
        # depth_image_tensor = torch.tensor([depth_image], device='cuda:0')

        # Display the normalized depth image
        # cv2.imshow("Cropped Depth Image", depth_image)

        # rospy.loginfo(depth_image_tensor)
        batch['depth'] = depth_image
        save_batch(batch)
        # cv2.waitKey(0) 
        # cv2.destropAllWindows()
    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))
    
    depth_image_sub.unregister()


def save_batch(batch):
    try:
        with open('/home/senirud/catkin_ws1/src/robot_controller/data/batch_data.txt', 'w') as f:
            f.write(str(batch))
            rospy.loginfo("Batch data saved successfully")
    except Exception as e:
        rospy.logerr("Error saving batch data: %s", str(e))

def take_image():

    global rgb_image_sub, depth_image_sub, batch

    processor = TextProcessor('/home/senirud/catkin_ws1/src/robot_controller/data/Vocab_file.txt', torch.device('cuda:0'))
    text = "Follow the hallway until you see a gold colored trash can. Wait in front of the middle elevator."
    batch = processor.process(text)
    #print(batch)
    rgb_image_sub = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, rgb_image_callback)
    depth_image_sub = rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, depth_image_callback)  

    rospy.init_node('image_taker')



if __name__ == '__main__':
    try:
        #rospy.init_node('image_taker')
        take_image()
    except rospy.ROSInterruptException:
        pass
