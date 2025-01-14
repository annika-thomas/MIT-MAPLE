#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This agent demonstrates how to structure your code and visualize camera data in 
an OpenCV window and control the robot with keyboard commands with pynput 
https://pypi.org/project/opencv-python/
https://pypi.org/project/pynput/

"""
import numpy as np
import csv
import carla
import cv2 as cv
import random
from math import radians
from pynput import keyboard
import os
import shutil
from src.utils import compute_blob_mean_and_covariance
import matplotlib.pyplot as plt
from src.utils import plotErrorEllipse
import skimage
from src.FastSAM.fastsam import *
from src.FastSamWrapper import FastSamWrapper


""" Import the AutonomousAgent from the Leaderboard. """

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

""" Define the entry point so that the Leaderboard can instantiate the agent class. """

def get_entry_point():
    return 'OpenCVagent'

""" Inherit the AutonomousAgent class. """

class OpenCVagent(AutonomousAgent):

    def setup(self, path_to_conf_file):

        """ This method is executed once by the Leaderboard at mission initialization. We should add any attributes to the class using 
        the 'self' Python keyword that contain data or methods we might need throughout the simulation. If you are using machine learning 
        models for processing sensor data or control, you should load the models here. We encourage the use of class attributes in place
        of using global variables which can cause conflicts. """

        """ Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys. """

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """

        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = 0

        self.columns = ['frame', 'power', 'input_v', 'input_w', 'gt_x', 'gt_y', 'gt_z', 'gt_roll', 'gt_pitch', 'gt_yaw', 'imu_accel_x', 'imu_accel_y', 'imu_accel_z', 'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']
        self.imu = []

        self.trial = '005'

        # FastSAM setup
        self.FastSamModel = FastSamWrapper('./dependencies/FastSAM/Models/FastSAM-x.pt', 'cuda', 0.5, 0.9)


    def use_fiducials(self):

        """ We want to use the fiducials, so we return True. """
        return True

    def sensors(self):

        """ In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048) 
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light. """

        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.Left: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.Right: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.BackRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
            carla.SensorPosition.Back: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720', 'use_semantic': True
            },
        }
        return sensors

    def run_step(self, input_data):

        """ The run_step method executes in every simulation time-step. Your control logic should go here. """

        """ In the first frame of the simulation we want to raise the robot's excavating arms to remove them from the 
        field of view of the cameras. Remember that we are working in radians. """

        if self.frame == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        """ Let's retrieve the front left camera data from the input_data dictionary using the correct dictionary key. We want the 
        grayscale monochromatic camera data, so we use the 'Grayscale' key. """

        sensor_data_frontleft = input_data['Grayscale'][carla.SensorPosition.FrontLeft]
        sensor_data_frontright = input_data['Grayscale'][carla.SensorPosition.FrontRight]
        sensor_data_left = input_data['Grayscale'][carla.SensorPosition.Left]
        sensor_data_right = input_data['Grayscale'][carla.SensorPosition.Right]
        sensor_data_backleft = input_data['Grayscale'][carla.SensorPosition.BackLeft]
        sensor_data_backright = input_data['Grayscale'][carla.SensorPosition.BackRight]
        sensor_data_front = input_data['Grayscale'][carla.SensorPosition.Front]
        sensor_data_back = input_data['Grayscale'][carla.SensorPosition.Back]

        # segment_masks = self.FastSamModel.segmentFrame(sensor_data_frontleft)
        # if (segment_masks is not None):
        #     print("len segment masks", len(segment_masks))
        # else:
        #     print("no segments found")

        if (sensor_data_frontleft is not None and self.frame%7==0):
            # running our fastSAM stuff here
            image_gray_rgb = np.stack((sensor_data_frontleft,)*3, axis=-1)
            segment_masks = self.FastSamModel.segmentFrame(image_gray_rgb)

            print("sensor_data_frontleft shape", sensor_data_frontleft.shape)

            

            print("image_gray_rgb shape", image_gray_rgb.shape)

            blob_means = []
            blob_covs = []

            # If there were segmentations detected by FastSAM, transfer them from GPU to CPU and convert to Numpy arrays
            # if (len(segment_masks) == 0):
            #     print("no segments returned")
            #     segment_masks = None
            fig, ax = plt.subplots(figsize=(8, 6))  # Define the figure and axis

            if (segment_masks is not None):
                print("number of segment masks returned: ", len(segment_masks))
                # FastSAM provides a numMask-channel image in shape C, H, W where each channel in the image is a binary mask
                # of the detected segment
                [numMasks, h, w] = segment_masks.shape

                print("segment mask shape:", numMasks, h, w)

                # Prepare a mask of IDs where each pixel value corresponds to the mask ID
                segment_masks_flat = np.zeros((h,w),dtype=int)

                for maskId in range(numMasks):
                    # Extract the single binary mask for this mask id
                    # print("masking id", maskId)
                    mask_this_id = segment_masks[maskId,:,:]

                    # From the pixel coordinates of the mask, compute centroid and a covariance matrix
                    blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
                    
                    # Store centroids and covariances in lists
                    blob_means.append(blob_mean)
                    blob_covs.append(blob_cov)

                    # Replace the 1s corresponding with masked areas with maskId (i.e. number of mask)
                    segment_masks_flat = np.where(mask_this_id < 1, segment_masks_flat, maskId)

                # Using skimage, overlay masked images with colors
                print("overlaying images")
                print("segmasks flat shape", segment_masks_flat.shape)
                print("image gray rgb shape", image_gray_rgb.shape)
                # image_gray_rgb = skimage.color.label2rgb(segment_masks_flat, image_gray_rgb)
                # Overlay masked images with colors
                image_gray_rgb = skimage.color.label2rgb(segment_masks_flat, image_gray_rgb)
                ax.imshow(image_gray_rgb)  # Display the image with overlaid masks

            # For each centroid and covariance, plot an ellipse
            for m, c in zip(blob_means, blob_covs):
                print("plotting error elipse at", m[0], m[1], c)
                plotErrorEllipse(ax, m[0], m[1], c, "r",stdMultiplier=2.0)

            # Show image using Matplotlib, set axis limits, save and close the output figure.
            # ax.imshow(image_gray_rgb)
            # Display the overlaid image on the Matplotlib axis
            ax.imshow(image_gray_rgb)
            ax.axis('off')  # Remove axis for a cleaner saved image

            # Save the Matplotlib figure as an image
            output_path = f'data/{self.trial}/FastSAM/{self.frame}_fastsam_means_covs.png'
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            print(f"Saved plot as {output_path}")

            print("trying to save image")
            # Normalize and convert to uint8
            image_to_save = (image_gray_rgb * 255).astype(np.uint8)

            # Save the image
            cv.imwrite(f'data/{self.trial}/FastSAM/' + str(self.frame) + '_fastsam.png', cv.cvtColor(image_to_save, cv.COLOR_RGB2BGR))


            # cv.imwrite(f'data/{self.trial}/FastSAM/' + str(self.frame) + '_fastsam.png', image_gray_rgb)
        # ax.set_xlim([0,w])
        # ax.set_ylim([h,0])
        # fig.savefig(filepath_out, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)




        imu_data = self.get_imu_data()
        mission_time = round(self.get_mission_time(), 2)
        vehicle_data=self._vehicle_status
        transform = vehicle_data.transform
        # current_power = self._controller.agent.get_current_power()
        # power_percent = self._controller.agent.get_current_power() / self._max_power*100.
        print("imu data: ", imu_data)

        transform_location_x = transform.location.x
        transform_location_y = transform.location.y
        transform_location_z = transform.location.z
        transform_rotation_r = transform.rotation.roll
        transform_rotation_p = transform.rotation.pitch
        transform_rotation_y = transform.rotation.yaw
        input_v = self.current_v
        input_w = self.current_w
        power = self.get_current_power()

        # self.imu.append([self.frame] + [transform.location.x, transform.location.y, transform.location.z] + imu_data)
        imu_entry = [self.frame] + \
            [power, input_v, input_w, transform_location_x, transform_location_y, transform_location_z, transform_rotation_r, transform_rotation_p, transform_rotation_y] + \
            imu_data.tolist()  # Convert NumPy array to list

        # Append to self.imu
        self.imu.append(imu_entry)

        """ We need to check that the sensor data is not None before we do anything with it. The date for each camera will be 
        None for every other simulation step, since the cameras operate at 10Hz while the simulator operates at 20Hz. """

        if sensor_data_frontleft is not None:

            cv.imshow('Left camera view', sensor_data_frontleft)
            cv.waitKey(1)
            dir_frontleft = f'data/{self.trial}/FrontLeft/'

            if not os.path.exists(dir_frontleft):
                os.makedirs(dir_frontleft)

            semantic = input_data['Semantic'][carla.SensorPosition.FrontLeft]
            cv.imwrite(dir_frontleft + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_frontleft + str(self.frame) + '.png', sensor_data_frontleft)
            print("saved image front left ", self.frame)

        if sensor_data_frontright is not None:

            dir_frontright = f'data/{self.trial}/FrontRight/'

            if not os.path.exists(dir_frontright):
                os.makedirs(dir_frontright)

            semantic = input_data['Semantic'][carla.SensorPosition.FrontRight]
            cv.imwrite(dir_frontright + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_frontright + str(self.frame) + '.png', sensor_data_frontright)
            print("saved image front right ", self.frame)

        if sensor_data_left is not None:

            dir_left = f'data/{self.trial}/Left/'

            if not os.path.exists(dir_left):
                os.makedirs(dir_left)

            semantic = input_data['Semantic'][carla.SensorPosition.Left]
            cv.imwrite(dir_left + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_left + str(self.frame) + '.png', sensor_data_left)
            print("saved image left ", self.frame)

        if sensor_data_right is not None:

            dir_right = f'data/{self.trial}/Right/'

            if not os.path.exists(dir_right):
                os.makedirs(dir_right)

            semantic = input_data['Semantic'][carla.SensorPosition.Right]
            cv.imwrite(dir_right + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_right + str(self.frame) + '.png', sensor_data_right)
            print("saved image right ", self.frame)

        if sensor_data_backleft is not None:

            dir_backleft = f'data/{self.trial}/BackLeft/'

            if not os.path.exists(dir_backleft):
                os.makedirs(dir_backleft)

            semantic = input_data['Semantic'][carla.SensorPosition.BackLeft]
            cv.imwrite(dir_backleft + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_backleft + str(self.frame) + '.png', sensor_data_backleft)
            print("saved image back left ", self.frame)

        if sensor_data_backright is not None:

            dir_backright = f'data/{self.trial}/BackRight/'

            if not os.path.exists(dir_backright):
                os.makedirs(dir_backright)

            semantic = input_data['Semantic'][carla.SensorPosition.BackRight]
            cv.imwrite(dir_backright + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_backright + str(self.frame) + '.png', sensor_data_backright)
            print("saved image back right ", self.frame)

        if sensor_data_front is not None:

            dir_front = f'data/{self.trial}/Front/'

            if not os.path.exists(dir_front):
                os.makedirs(dir_front)

            semantic = input_data['Semantic'][carla.SensorPosition.Front]
            cv.imwrite(dir_front + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_front + str(self.frame) + '.png', sensor_data_front)
            print("saved image front ", self.frame)

        if sensor_data_back is not None:

            dir_back = f'data/{self.trial}/Back/'

            if not os.path.exists(dir_back):
                os.makedirs(dir_back)

            semantic = input_data['Semantic'][carla.SensorPosition.Back]
            cv.imwrite(dir_back + str(self.frame) + '_sem.png', semantic)

            cv.imwrite(dir_back + str(self.frame) + '.png', sensor_data_back)
            print("saved image back ", self.frame)



        """ Now we prepare the control instruction to return to the simulator, with our target linear and angular velocity. """

        self.frame += 1

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        
        """ If the simulation has been going for more than 5000 frames, let's stop it. """
        if self.frame >= 5000:
            self.mission_complete()

        return control

    def finalize(self):

        """ In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources. 
        In this case, we should close the OpenCV window. """

        # Save the data to a CSV file
        output_filename_imu = f"/home/annikat/LAC/LunarAutonomyChallenge/data/{self.trial}/imu_data.csv"

        # Write to CSV file
        with open(output_filename_imu, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)  # Write header
            writer.writerows(self.imu)    # Write the IMU data rows

        print(f"Data saved to {output_filename_imu}")

        cv.destroyAllWindows()

        """ We may also want to add any final updates we have from our mapping data before the mission ends. Let's add some random values 
        to the geometric map to demonstrate how to use the geometric map API. The geometric map should also be updated during the mission
        in the run_step() method, in case the mission is terminated unexpectedly. """

        """ Retrieve a reference to the geometric map object. """

        geometric_map = self.get_geometric_map()

        map_array = self.get_map_array()

        print("Map array:", map_array)

        # # Save the data to a CSV file
        # output_filename_map_gt = f"/home/annikat/LAC/LunarAutonomyChallenge/data/{self.trial}/map_gt.csv"

        # np.savetxt(output_filename_map_gt, map_array, delimiter=",", fmt="%d")

        # print(f"Map saved to {output_filename_map_gt}")

        """ Set some random height values and rock flags. """

        for i in range(100):

            x = 10 * random.random() - 5
            y = 10 * random.random() - 5
            geometric_map.set_height(x, y, random.random())

            rock_flag = random.random() > 0.5
            geometric_map.set_rock(x, y, rock_flag)

        map_array = self.get_map_array()

        # print("Map array:", map_array)

        # # Save the data to a CSV file
        # output_filename_map_gt = f"/home/annikat/LAC/LunarAutonomyChallenge/data/{self.trial}/map_gt.csv"

        # np.savetxt(output_filename_map_gt, map_array, delimiter=",", fmt="%d")

        # print(f"Map saved to {output_filename_map_gt}")


    def on_press(self, key):

        """ This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular 
        velocity of 0.6 radians per second. """

        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6      

    def on_release(self, key):

        """ This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot. """

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()

