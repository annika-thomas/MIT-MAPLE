import numpy as np
import cv2
import json

class IPExStereoDepthMapper:
    def __init__(self, image_width=1280, image_height=720):
        """
        Initialize the stereo depth mapper for IPEx robot's forward stereo cameras
        
        Parameters:
        image_width: int, width of input images
        image_height: int, height of input images
        """
        self.width = image_width
        self.height = image_height
        
        # Camera parameters
        self.fov = 1.22  # 70 degrees in radians
        self.baseline = 0.162  # meters
        
        # Calculate focal length in pixels
        self.focal_length_pixels = (self.width / 2) / np.tan(self.fov / 2)
        
        # Camera positions in robot coordinates
        self.left_camera_pos = np.array([0.28, 0.081, 0.131])
        self.right_camera_pos = np.array([0.28, -0.081, 0.131])
        self.camera_orientation = np.array([1, 0, 0])  # Unit vector
        
        # Initialize stereo matcher
        self.window_size = 11
        self.min_disp = 0
        self.num_disp = 128  # Adjust based on your scene depth range
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            disp12MaxDiff=1,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
        )

        self.keyframe_data = {}


    def save_keyframe(self, frame_number, boulder_positions):
        """
        Save boulder positions for a specific keyframe
        
        Parameters:
        frame_number: int, the frame number to save
        boulder_positions: list of (x,y,z) positions in robot coordinates
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_positions = []
        for pos in boulder_positions:
            if pos is not None:
                if isinstance(pos, np.ndarray):
                    serializable_positions.append(pos.tolist())
                else:
                    serializable_positions.append(pos)
            else:
                serializable_positions.append(None)
                
        self.keyframe_data[str(frame_number)] = serializable_positions
    
    def save_checkpoint(self, filepath):
        """
        Save the keyframe data to a JSON file
        
        Parameters:
        filepath: str, path to save the checkpoint file
        """
        with open(filepath, 'w') as f:
            json.dump(self.keyframe_data, f)
            
    def load_checkpoint(self, filepath):
        """
        Load keyframe data from a JSON file
        
        Parameters:
        filepath: str, path to the checkpoint file
        """
        with open(filepath, 'r') as f:
            self.keyframe_data = json.load(f)
            
    def get_keyframe_data(self, frame_number):
        """
        Get boulder positions for a specific keyframe
        
        Parameters:
        frame_number: int, the frame number to retrieve
        
        Returns:
        list of (x,y,z) positions, or None if frame not found
        """
        return self.keyframe_data.get(str(frame_number))

    def compute_depth_map(self, left_img, right_img):
        """
        Compute depth map from stereo images
        
        Parameters:
        left_img, right_img: Grayscale images (720, 1280) from left and right cameras
        
        Returns:
        depth_map: numpy array containing depth values in meters for each pixel
        confidence_map: numpy array containing confidence values (0-1) for each depth estimate
        """
        # Verify image dimensions
        if left_img.shape != (self.height, self.width) or right_img.shape != (self.height, self.width):
            raise ValueError(f"Expected image size {self.width}x{self.height}, got {left_img.shape}")
            
        # Verify images are grayscale
        if len(left_img.shape) != 2 or len(right_img.shape) != 2:
            raise ValueError("Input images must be grayscale (2D arrays)")
        
        # Compute disparity map
        disparity = self.stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        
        # Calculate depth map
        depth_map = np.zeros_like(disparity)
        valid_disparity = disparity > 0
        
        # Z = baseline * focal_length / disparity
        depth_map[valid_disparity] = (self.baseline * self.focal_length_pixels) / disparity[valid_disparity]
        
        # Computing confidence based on disparity and texture
        confidence_map = np.zeros_like(disparity)
        
        # Higher confidence for:
        # 1. Stronger disparity values
        # 2. Areas with good texture (using Sobel gradient magnitude)
        gradient_x = cv2.Sobel(left_img, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(left_img, cv2.CV_64F, 0, 1, ksize=3)
        texture_strength = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize texture strength to 0-1
        texture_strength = cv2.normalize(texture_strength, None, 0, 1, cv2.NORM_MINMAX)
        
        # Combine disparity confidence and texture confidence
        disp_confidence = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX)
        confidence_map = 0.7 * disp_confidence + 0.3 * texture_strength
        confidence_map[~valid_disparity] = 0
        
        return depth_map, confidence_map

    
    def get_object_positions(self, depth_map, centroids, window_size=5):
        """
        Get 3D positions of objects in robot coordinates using their image centroids
        
        Parameters:
        depth_map: numpy array of depth values
        centroids: list of (x,y) centroids in image coordinates
        window_size: size of window to average depth values around centroid
        
        Returns:
        positions: list of (x,y,z) positions in robot coordinates
        depths: list of depth values used for each object
        """
        positions = []
        depths = []
        
        for centroid in centroids:
            print("centroid", centroid)
            x, y = int(centroid[0]), int(centroid[1])
            
            # Get average depth in window around centroid
            half_window = window_size // 2
            y_start = max(0, y - half_window)
            y_end = min(depth_map.shape[0], y + half_window + 1)
            x_start = max(0, x - half_window)
            x_end = min(depth_map.shape[1], x + half_window + 1)
            
            depth_window = depth_map[y_start:y_end, x_start:x_end]
            valid_depths = depth_window[depth_window > 0]
            
            if len(valid_depths) > 0:
                # Use median depth to be robust to outliers
                depth = np.median(valid_depths)
                
                # Convert image coordinates to normalized camera coordinates
                x_norm = (x - self.width/2) / self.focal_length_pixels
                y_norm = (y - self.height/2) / self.focal_length_pixels
                
                # Get 3D point in camera coordinates
                point = np.array([
                    depth * x_norm,  # X = Z * (x-cx)/fx
                    depth * y_norm,  # Y = Z * (y-cy)/fy
                    depth           # Z = depth
                ])
                
                # Transform to robot coordinates
                point = point @ np.array([
                    [1, 0, 0],  # Camera faces positive X
                    [0, 1, 0],
                    [0, 0, 1]
                ])
                
                # Translate to robot coordinates
                point += self.left_camera_pos
                
                positions.append(point)
                depths.append(depth)
            else:
                # If no valid depth found, append None
                positions.append(None)
                depths.append(None)
        
        return positions, depths

    def get_3d_point_cloud(self, depth_map):
        """
        Convert depth map to 3D point cloud in robot coordinates
        
        Parameters:
        depth_map: numpy array of depth values
        
        Returns:
        points: Nx3 numpy array of 3D points in robot coordinates
        """
        rows, cols = depth_map.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Convert image coordinates to normalized camera coordinates
        x = (c - cols/2) / self.focal_length_pixels
        y = (r - rows/2) / self.focal_length_pixels
        
        # Scale by depth to get 3D points in camera coordinates
        points = np.stack([
            depth_map * x,
            depth_map * y,
            depth_map
        ], axis=-1)
        
        # Transform to robot coordinates (from left camera frame)
        # Rotate points according to camera orientation
        points = points @ np.array([
            [1, 0, 0],  # Camera faces positive X
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Translate to robot coordinates
        points += self.left_camera_pos
        
        return points[depth_map > 0]  # Return only valid points