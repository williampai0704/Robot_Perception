import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    """Check if array is a valid transform.

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.

    We provide this function. Students don't need to implment this function. 
    """
    # check shape
    if t.shape != (4,4):
        return False

    # check all elements are real
    real_check = np.all(np.isreal(t))

    # calc intermediates
    rtr = np.matmul(t[:3, :3].T, t[:3, :3])
    rrt = np.matmul(t[:3, :3], t[:3, :3].T)

    # make rtr and rrt are identity
    inverse_check = np.isclose(np.eye(3), rtr, atol=tolerance).all() and np.isclose(np.eye(3), rrt, atol=tolerance).all()

    # check det
    det_check = np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=tolerance).all()

    # make sure last row is correct
    last_row_check = np.isclose(t[3, :3], np.zeros((1, 3)), atol=tolerance).all() and np.isclose(t[3, 3], 1.0, atol=tolerance).all()

    return real_check and inverse_check and det_check and last_row_check

def transform_concat(t1, t2):
    """[summary]

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    # TODO: 
    if not transform_is_valid(t1):
        raise ValueError('t1 is invalid.')
    if not transform_is_valid(t2):
        raise ValueError('t2 is invalid.')
    return t1@t2
    

def transform_point3s(t, ps):
    """Transfrom 3D points from one space to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """

    # TODO: 
    if not transform_is_valid(t):
        raise ValueError('t is invalid.')
    row , col = ps.shape
    if int(col) != 3:
        raise ValueError('ps does not have correct shape')
    cc = np.ones((row,1))
    ps = np.column_stack((ps,cc))
    
    return (t@ps.T).T[:,:3]
    

def transform_inverse(t):
    """Find the inverse of the transfom.

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    # TODO:
    if not transform_is_valid(t):
        raise ValueError('t is invalid.')
    return np.linalg.inv(t)
    

def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.

    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.

    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point.
    """
    # TODO
    p = []
    row, col = depth_image.shape
    for v in range(row):
        for u in range(col):
            z = depth_image[v,u]
            if z > 0:
                x = (u - intrinsics[0,2]) / intrinsics[0,0]
                y = (v - intrinsics[1,2])/ intrinsics[1,1]
                p.append([x*z,y*z,z])
                
    return np.array(p)
    

def write_ply(ply_path, points, colors= np.array([])):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
            points (float): Nx3 x,y,z locations for each point 
            colors (uchar): Nx3 r,g,b color for each point 
            
        We provide this function for you. 
        """
        with open(ply_path, 'w') as f:
            # Write header.
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(len(points)))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')

            if len(colors) != 0:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')

            f.write('end_header\n')

            # Write points.
            for i in range(len(points)):
                f.write('{0} {1} {2}'.format(
                    points[i][0],
                    points[i][1],
                    points[i][2]))

                if len(colors) != 0:
                    f.write(' {0} {1} {2}'.format(
                        int(colors[i][0]),
                        int(colors[i][1]),
                        int(colors[i][2])))

                f.write('\n')

            