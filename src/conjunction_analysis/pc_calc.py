"""
Implement methods to calculate Pc given two satellites states using the following method:
https://www.space-track.org/documents/How_the_JSpOC_Calculates_Probability_of_Collision.pdf
https://stacks.stanford.edu/file/druid:dg552pb6632/Foster-estes-parametric_analysis_of_orbital_debris_collision_probability.pdf

"""
import numpy as np
import scipy.integrate as integrate
from numpy import linalg as LA

def gcrs2ric(rGCRS,vGCRS):
    """
    Get rotation matrix from GCRS (ECI) frame to RIC frame
    """
    # Get RIC from ICRS
    r_dir = rGCRS / np.linalg.norm(rGCRS)
    c_dir = np.cross(rGCRS,vGCRS) / np.linalg.norm(np.cross(rGCRS,vGCRS))
    i_dir = np.cross(c_dir,r_dir)

    # Build rotation matrix
    rot_GCRS2RIC = np.array([r_dir,i_dir,c_dir])

    return rot_GCRS2RIC

def norm_vector(v):
    norm = LA.norm(v)
    if norm == 0:
        return v*0
    else:
        return v / norm

class satellite():
    def __init__(self, id, size, pos, vel, cov_rtn):
        self.id = id
        self.size = size
        self.pos = pos
        self.vel = vel

        # Rotation matrix GCRS to RIC
        rot_GCRS2RIC = gcrs2ric(self.pos,self.vel)

        # Get covariance matrix in GCRS frame
        self.cov = rot_GCRS2RIC.T @ cov_rtn @ rot_GCRS2RIC

class calcPC():
    def __init__(self, sat1, sat2):
        # self.Pc = self.get_pc(sat1=sat1, sat2=sat2)
        d, C, r_sp = self.get2DParams(sat1=sat1, sat2=sat2)
        self.Pc = self.integratePC(d, C, r_sp)

    def get2DParams(self, sat1, sat2):
        # Get key values
        d = sat1.size + sat2.size
        r_rel = sat2.pos - sat1.pos
        v_rel = sat2.vel - sat1.vel
        r_sp = np.array([LA.norm(r_rel), 0])
        C3D = sat1.cov + sat2.cov

        # Find axes of conjunction frame
        x_hat = norm_vector(r_rel)
        y_hat = np.cross(x_hat, norm_vector(v_rel))
        z_hat = np.cross(x_hat, y_hat)

        # # Construct 3D intertial to conjunction matrix
        # V = np.vstack((x_hat, y_hat, z_hat)).T
        # # TODO: Add check if V is nonsingular
        # R = V @ LA.inv(V)

        # # Convert values to covariance matrix form
        # C3D_conj = R @ C3D
        # eigenvalues, eigenvectors = LA.eig(C3D_conj)

        # # Project 3D covariance into 2D conjunction plane
        # v1, v2, v3 = [eigenvalues[0]*eigenvectors[:,0], eigenvalues[1]*eigenvectors[:,1], eigenvalues[2]*eigenvectors[:,2]]
        # p1 = self.project_onto_basis(v1)
        # p2 = self.project_onto_basis(v2)
        # p3 = self.project_onto_basis(v3)
        # V, Lambda = self.get_2D_eigenmats(p1, p2, p3)

        # C = V @ Lambda @ LA.inv(V)

        # P = np.eye(3) - z_hat @ z_hat.T
        P = np.vstack([x_hat, y_hat]).T
        C = P.T @ C3D @ P
        # C = C3D - np.dot(C3D, z_hat) * z_hat

        return d, C, r_sp

    # def get_pc(self, sat1, sat2):
    #     # Get key parameters
    #     R = sat1.size + sat2.size
    #     v_r = sat2.vel - sat1.vel
    #     r_0 = sat2.pos - sat1.pos
    #     U_hat = norm_vector(np.cross(sat1.vel, sat2.vel))
    #     V_hat = norm_vector(v_r)
    #     W_hat = np.cross(U_hat, V_hat)
    #     theta_r = np.arccos(np.clip(np.dot(sat2.vel, sat1.vel), -1.0, 1.0)) / 2
    #     t_cpa = -np.dot(r_0, v_r) / np.dot(v_r, v_r)

    #     sigma_u = np.sqrt(sat1.cov[0,0]**2 + sat2.cov[0,0]**2)
    #     sigma_w = np.sqrt((sat1.cov[1,1]*np.sin(theta_r))**2 +
    #                       (sat1.cov[2,2]*np.cos(theta_r))**2 +
    #                       (sat2.cov[1,1]*np.sin(theta_r))**2 +
    #                       (sat2.cov[2,2]*np.cos(theta_r))**2)
        
    #     U_0 = np.dot(r_0, U_hat)
    #     W_0 = np.dot(r_0, W_hat)

    #     f = lambda r, t: np.exp(-((r*np.cos(t) - W_0)**2) / (2*sigma_w**2)) * np.exp(-((r*np.sin(t) - U_0)**2) / (2*sigma_u**2))
    #     integ = integrate.dblquad(lambda r, theta: f(r, theta) * r,
    #                               0, R,
    #                               lambda theta: 0, lambda theta: 2*np.pi)
    #     norm = 1 / (2*np.pi*sigma_u*sigma_w)
    #     return integ[0] * norm


    
    @staticmethod
    def project_onto_basis(v):
        uz = np.array([0, 0, 1])
        v_perp = np.dot(v, uz) * uz
        v_plane = v - v_perp
        return v_plane[:-1]

    @staticmethod
    def get_2D_eigenmats(p1, p2, p3):
        x = [p1[0], p2[0], p3[0]]
        y = [p1[1], p2[1], p3[1]]
        indZero = -1
        for i in range(3):
            if np.sqrt(x[i]**2 + y[i]**2) < 1e-6:
                indZero = i
        
        # If ellipse is already aligned with plane
        if indZero != -1:
            del x[indZero]
            del y[indZero]
            V = np.eye(2)
            Lambda = np.diag(np.sqrt(x[0]**2 + y[0]**2), np.sqrt(x[1]**2 + y[1]**2))
            return V, Lambda
        
        X = np.array([[x[0]**2, y[0]**2, 2*x[0]*y[0]],
                      [x[1]**2, y[1]**2, 2*x[1]*y[1]],
                      [x[2]**2, y[2]**2, 2*x[2]*y[2]]])
        Constants = LA.inv(X) @ np.array([1, 1, 1])
        a, b, c = Constants
        a = np.real(a)
        b = np.real(b)
        c = np.real(c)
        theta = 0.5*np.arctan2(2*c, b-a)
        eta = a + b
        zeta = b - a / np.cos(2*theta)
        ra = np.sqrt(2/np.abs(eta-zeta))
        rp = np.sqrt(2/(eta+zeta))
        ang = 2*np.pi - theta
        V = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
        Lambda = np.diag([ra, rp])
        return V, Lambda

    def integratePC(self, d, C, r_sp):
        f = lambda x, y: np.exp(-0.5 * ((np.array([x, y]) - r_sp).T @ LA.inv(C) @ (np.array([x, y]) - r_sp)))
        integ = integrate.dblquad(f, -d, d, lambda x: -np.sqrt(d**2 - x**2), lambda x: np.sqrt(d**2 - x**2))
        norm = 1 / (2*np.pi*np.sqrt(LA.det(C)))
        return integ[0]*norm