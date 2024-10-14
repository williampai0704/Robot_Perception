
from skimage import measure
from transforms import *


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, volume_bounds, voxel_size):
        """Initialize tsdf volume instance variables.

        Args:
            volume_bounds (numpy.array [3, 2]): rows index [x, y, z] and cols index [min_bound, max_bound].
                Note: units are in meters.
            voxel_size (float): The side length of each voxel in meters.

        Raises:
            ValueError: If volume bounds are not the correct shape.
            ValueError: If voxel size is not positive.
        """
        volume_bounds = np.asarray(volume_bounds)
        if volume_bounds.shape != (3, 2):
            raise ValueError('volume_bounds should be of shape (3, 2).')

        if voxel_size <= 0.0:
            raise ValueError('voxel size must be positive.')

        # Define voxel volume parameters
        self._volume_bounds = volume_bounds
        self._voxel_size = float(voxel_size)
        self._truncation_margin = 2 * self._voxel_size  # truncation on SDF (max alowable distance away from a surface)

        # Adjust volume bounds and ensure C-order contiguous
        # and calculate voxel bounds taking the voxel size into consideration
        self._voxel_bounds = np.ceil(
            (self._volume_bounds[:, 1] - self._volume_bounds[:, 0]) / self._voxel_size
        ).copy(order='C').astype(int)
        self._volume_bounds[:, 1] = self._volume_bounds[:, 0] + self._voxel_bounds * self._voxel_size

        # volume min bound is the origin of the volume in world coordinates
        self._volume_origin = self._volume_bounds[:, 0].copy(order='C').astype(np.float32)

        print('Voxel volume size: {} x {} x {} - # voxels: {:,}'.format(
            self._voxel_bounds[0],
            self._voxel_bounds[1],
            self._voxel_bounds[2],
            self._voxel_bounds[0] * self._voxel_bounds[1] * self._voxel_bounds[2]))

        # Initialize pointers to voxel volume in memory
        self._tsdf_volume = np.ones(self._voxel_bounds).astype(np.float32)

        # for computing the cumulative moving average of observations per voxel
        #self._weight_volume = np.zeros(self._voxel_bounds).astype(np.float32)
        color_bounds = np.append(self._voxel_bounds, 3)
        self._color_volume = np.zeros(color_bounds).astype(np.float32)  # rgb order

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._voxel_bounds[0]),
            range(self._voxel_bounds[1]),
            range(self._voxel_bounds[2]),
            indexing='ij')
        self._voxel_coords = np.concatenate([
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)], axis=0).astype(int).T

    def get_volume(self):
        """Get the tsdf and color volumes.

        Returns:
            numpy.array [l, w, h]: l, w, h are the dimensions of the voxel grid in voxel space.
                Each entry contains the integrated tsdf value.
            numpy.array [l, w, h, 3]: l, w, h are the dimensions of the voxel grid in voxel space.
                3 is the channel number in the order r, g, then b.
        """
        return self._tsdf_volume, self._color_volume

    def get_mesh(self):
        """ Run marching cubes over the constructed tsdf volume to get a mesh representation.

        Returns:
            numpy.array [n, 3]: each row represents a 3D point.
            numpy.array [k, 3]: each row is a list of point indices used to render triangles.
            numpy.array [n, 3]: each row represents the normal vector for the corresponding 3D point.
            numpy.array [n, 3]: each row represents the color of the corresponding 3D point.
        """
        tsdf_volume, color_vol = self.get_volume()

        # Marching cubes
        voxel_points, triangles, normals, _ = measure.marching_cubes(tsdf_volume, level=0, method='lewiner')
        points_ind = np.round(voxel_points).astype(int)
        points = self.voxel_to_world(self._volume_origin, voxel_points, self._voxel_size)

        # Get vertex colors.
        rgb_vals = color_vol[points_ind[:, 0], points_ind[:, 1], points_ind[:, 2]]
        colors_r = rgb_vals[:, 0]
        colors_g = rgb_vals[:, 1]
        colors_b = rgb_vals[:, 2]
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        return points, triangles, normals, colors

    def get_valid_points(self, depth_image, voxel_u, voxel_v, voxel_z):
        """ Compute a boolean array for indexing the voxel volume and other variables.
        Note that every time the method integrate(...) is called, not every voxel in
        the volume will be updated. This method returns a boolean matrix called
        valid_points with dimension (n, ), where n = # of voxels. Index i of
        valid_points will be true if this voxel will be updated, false if the voxel
        needs not to be updated.

        The criteria for checking if a voxel is valid or not is shown below.

        Args:
            depth_image (numpy.array [h, w]): A z depth image.
            voxel_u (numpy.array [v, ]): Voxel coordinate projected into camera coordinate, axis is u
            voxel_v (numpy.array [v, ]): Voxel coordinate projected into camera coordinate, axis is v
            voxel_z (numpy.array [v, ]): Voxel coordinate projected into world coordinate axis z
        Returns:
            valid_points numpy.array [v, ]: A boolean matrix that will be
            used to index into the voxel grid. Note the dimension of this
            variable.
        """

        image_height, image_width = depth_image.shape

        #  Eliminate pixels not in the image bounds or that are behind the image plane
        valid_pixels = np.logical_and(voxel_u >= 0,
                                np.logical_and(voxel_u < image_width,
                                np.logical_and(voxel_v >= 0,
                                np.logical_and(voxel_v < image_height,
                                voxel_z > 0))))


        #  Get depths for valid coordinates u, v from the depth image. Zero elsewhere.
        depths = np.zeros(voxel_u.shape)
        depths[valid_pixels] = depth_image[voxel_v[valid_pixels], voxel_u[valid_pixels]]


        #  Filter out zero depth values and cases where depth + truncation margin >= voxel_z
        valid_points = depths > 0

        return valid_points

    """
    *******************************************************************************
    ****************************** ASSIGNMENT BEGINS ******************************
    *******************************************************************************
    """

    @staticmethod
    @njit(parallel=True)
    def voxel_to_world(volume_origin, voxel_coords, voxel_size):
        """ Convert from voxel coordinates to world coordinates
            (in effect scaling voxel_coords by voxel_size).

        Args:
            volume_origin (numpy.array [3, ]): The origin of the voxel
                grid in world coordinate space.
            voxel_coords (numpy.array [n, 3]): Each row gives the 3D coordinates of a voxel.
            voxel_size (float): The side length of each voxel in meters.

        Returns:
            numpy.array [n, 3]: World coordinate representation of each of the n 3D points.
        """
        
        # TODO
        return voxel_coords*voxel_size + volume_origin.T 
    
    @staticmethod
    def transfrom_wrd2cam(camera_pose, voxel_world_coords):
        # transform the points from the world frame to camera frame.
        # Args: 
        #   camera_pose [4x4]
        #   voxel_world_coords [nx3]
        # Returns: 
        #   voxel_cam_coords [nx3]

        # TODO
        v_w_homogeneous = np.hstack([voxel_world_coords, np.ones((len(voxel_world_coords), 1), dtype=np.float32)])
        return (np.linalg.inv(camera_pose)@v_w_homogeneous.T).T[:,:3]

    @staticmethod
    def compute_tsdf(voxel_z, valid_points, depth_image, valid_pixels, truncation_margin): 
        # Compute the new TSDF value for each valid point. 
        # remember to apply truncation and normalization in the end, so that tsdf value is in range [-1,1]
        # Args: 
        #   voxel_z: [m,1] depth value for all voxels, m: voxel size 
        #   valid_points: index of voxels that are valid 
        #   depth_image: HxW depth image 
        #   valid_pixels: [n, 2] 2D coordinate for all valid voxels
        #   truncation_margin: truncation_margin
        # Returns:  
        #   tsdf_new: [n,1] TSDF value for each valid point
        # TODO
        
        tsdf = np.zeros((valid_pixels.shape[0],))
        depth_values = depth_image[valid_pixels[:, 1], valid_pixels[:, 0]]
        tsdf = np.clip((depth_values - voxel_z[valid_points]) / truncation_margin, -1, 1)
        
        return tsdf  
            

    @staticmethod
    # @njit(parallel=True)
    def update_tsdf(tsdf_old, tsdf_new, color_old, color_new):
        """[summary]

        Args:
            tsdf_old (numpy.array [v, ]): v is equal to the number of voxels to be
                integrated at this timestamp. Old tsdf values that need to be
                updated based on the current observation.
            margin_distance (numpy.array [v, ]): The tsdf values of the current observation.
                It should be of type numpy.array [v, ], where v is the number
                of valid voxels.
            color_old (numpy.array [n, 3]): Old colors from self._color_volume in RGB.
            color_new (numpy.array [n, 3]): Newly observed colors from the image in RGB
        
        Returns:
            tsdf_new numpy.array [v, ]: new tsdf values for entries in tsdf_old
            color_new numpy.array [v, ]: new color values for entries in tsdf_old
        
        # Hints:
            Only update the tsdf and color value when 
            the new absolute value of tsdf_new[i] is smaller than that of tsdf_old[i]
        """

        #TODO
        update = np.abs(tsdf_new) < np.abs(tsdf_old)
        tsdf_new = np.where(update,tsdf_new,tsdf_old)
        color_new = np.where(update[:,np.newaxis],color_new,color_old)
        return tsdf_new, color_new
    
    def integrate(
            self,
            color_image,
            depth_image,
            camera_intrinsics,
            camera_pose,
            observation_weight=1.0,
    ):
        """Integrate an RGB-D observation into the TSDF volume, by updating the weight volume,
            tsdf volume, and color volume.

        Args:
            color_image (numpy.array [h, w, 3]): An rgb image.
            depth_image (numpy.array [h, w]): A z depth image.
            camera_intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
            camera_pose (numpy.array [4, 4]): SE3 transform representing pose (camera to world)
            observation_weight (float, optional):  The weight to assign for the current
                observation. Defaults to 1.
        """
        color_image = color_image.astype(np.float32)

        #  Step 1: Scaling the voxel grid coordinates to the world
        #  space by calling `voxel_to_world`. 
        voxel_world_coords = self.voxel_to_world(
            self._volume_origin, self._voxel_coords, self._voxel_size
        )


        # Step 2: TODO
        # transform the points in world frame to camera frame.
        # Save the voxel z coordinate (depth of the voxels in the camera frame) for later use. 
        voxel_camera_coords = self.transfrom_wrd2cam(camera_pose, voxel_world_coords)
        voxel_z = voxel_camera_coords[:, 2]

        # Step 3:  
        #  Project 3D voxels in camera frame (voxel_camera_coords) to 2D image coordinate
        voxel_image_coords = camera_to_image(camera_intrinsics, voxel_camera_coords)

        # Step 4: 
        #  Get all of the valid points in the voxel grid by calling
        #  the helper get_valid_points.  
        valid_points = self.get_valid_points(
            depth_image, voxel_image_coords[:, 0], voxel_image_coords[:, 1], voxel_z
        )

        # Step 5:
        #  With the valid_points array as your indexing array, index into
        #  the self._voxel_coords variable to get the valid voxel x, y, and z.
        valid_voxels = self._voxel_coords[valid_points]

        # Step 6: With the valid_points array as your indexing array,
        #  get the valid pixels. Use those valid pixels to index into
        #  the depth_image, and find the valid margin distance.
        valid_pixels = voxel_image_coords[valid_points]
        

        # Step 7: TODO
        # Compute the new TSDF value for each valid point. 
        # remember to apply truncation and normalization in the end,
        # so that tsdf value is in range [-1,1]
        tsdf_new = self.compute_tsdf(
            voxel_z, valid_points, depth_image, valid_pixels, self._truncation_margin
        )
        

        
        # Step 8. TODO:
        #  Update the TSDF value and color for the voxels has new observations. 
        #  Only update the tsdf and color value when 
        #  the new absolute value of tsdf_new[i] is smaller than that of tsdf_old[i]
        tsdf_old  = self._tsdf_volume[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]]
        color_old = self._color_volume[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]]
        color_new = color_image[valid_pixels[:, 1], valid_pixels[:, 0]]
        
        tsdf_updated, color_updated = self.update_tsdf(tsdf_old, tsdf_new, color_old, color_new )
        
        self._tsdf_volume[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]] = tsdf_updated
        self._color_volume[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]] = color_updated

    """
    *******************************************************************************
    ******************************* ASSIGNMENT ENDS *******************************
    *******************************************************************************
    """
