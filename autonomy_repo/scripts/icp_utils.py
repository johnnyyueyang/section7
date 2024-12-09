import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def best_fit_transform(A: np.ndarray, B: np.ndarray):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Naxm numpy array of corresponding points
      B: Nbxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m - 1, :] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.eye(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src: np.ndarray, dst: np.ndarray, radius=0.01):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    # Use the NearestNeighbors class from sklearn to find the nearest neighbors
    neigh = NearestNeighbors(n_neighbors=1, radius=radius)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(
    A: np.ndarray, 
    B: np.ndarray, 
    init_pose=None, 
    max_iterations=20, 
    tolerance=0.001, 
    knn_radius=0.01
) -> np.ndarray:
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
    '''
    # Import tqdm for progress bar
    from tqdm import trange

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    for i in trange(max_iterations, desc="ICP Iterations"):
        # Find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T, radius=knn_radius)
        
        # Remove pairs with no neighbors within the radius
        valid = distances < np.inf
        if not np.any(valid):
            print("No valid correspondences found.")
            break
        src_matched = src[:m, valid].T
        dst_matched = dst[:m, indices[valid]].T

        # Compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src_matched, dst_matched)

        # Update the current source
        src = np.dot(T, src)

        # Check error
        mean_error = np.mean(distances[valid])
        print(f"Iteration {i+1}: mean error = {mean_error:.6f}")
        if np.abs(prev_error - mean_error) < tolerance:
            print("Convergence reached.")
            break
        prev_error = mean_error

    # Calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T

def open3d_icp(source, target, T_init):
    o3d.geometry.PointCloud.estimate_normals(
        source,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)
    )

    o3d.geometry.PointCloud.estimate_normals(
        target,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)
    )

    # Perform point-to-plane ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 1.0, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
    return reg_p2p.transformation
