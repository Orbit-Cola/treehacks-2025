"""
Implement methods to calculate Pc given two satellites states using the following method:
https://www.space-track.org/documents/How_the_JSpOC_Calculates_Probability_of_Collision.pdf

"""
import numpy as np
import scipy.integrate as integrate
from numpy import linalg as LA

def norm_vector(v):
    norm = LA.norm(v)
    if norm == 0:
        return v*0
    else:
        return v / norm
    
class satellite():
    def __init__(self, name):
        # TODO: implement code to get the satellite information in a class format
        self.size = ""
        self.pos = ""
        self.vel = ""
        self.cov = ""

class calcPC():

    def __init__(self, sat1, sat2):
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

        # Construct 3D intertial to conjunction matrix
        V = np.vstack((x_hat, y_hat, z_hat))
        # TODO: Add check if V is nonsingular
        R = V @ LA.inv(V)

        # Convert values to covariance matrix form
        C3D_conj = R @ C3D
        eigenvalues, eigenvectors = LA.eig(C3D_conj)

        # Project 3D covariance into 2D conjunction plane
        v1, v2, v3 = [eigenvalues[0]*eigenvectors[:,0], eigenvalues[1]*eigenvectors[:,1], eigenvalues[2]*eigenvectors[:,2]]
        p1 = self.project_onto_basis(v1)
        p2 = self.project_onto_basis(v2)
        p3 = self.project_onto_basis(v3)
        V, Lambda = self.get_2D_eigenmats(p1, p2, p3)

        C = V @ Lambda @ LA.inv(V)

        return d, C, r_sp
    
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
        theta = 0.5*np.arctan2(2*c, b-a)
        eta = a + b
        zeta = (b - a) / np.cos(2*theta)
        ra = np.sqrt(2/(eta-zeta))
        rp = np.sqrt(2/(eta+zeta))
        ang = 2*np.pi - theta
        V = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
        Lambda = np.diag([ra, rp])
        return V, Lambda


    @staticmethod
    def integratePC(d, C, r_sp):
        f = lambda x, y: np.exp(-0.5 * ((np.array([x, y]) - r_sp).T @ LA.inv(C) @ (np.array([x, y]) - r_sp)))
        int = integrate.dblquad(f, -d, d, lambda x: -np.sqrt(d**2 - x), lambda x: np.sqrt(d**2 - x))
        norm = 1 / (2*np.pi*np.sqrt(LA.det(C)))
        return norm*int[0]