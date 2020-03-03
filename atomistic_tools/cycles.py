import numpy as np

import scipy as sp
import scipy.linalg

import ase
import ase.io
import ase.visualize
import ase.neighborlist

import os
import shutil


def convert_neighbor_list(nl):
    new = {}
    n_vert = np.max(nl)+1
    
    for i_v in range(n_vert):
        new[i_v] = []
    
    for i_v, j_v in zip(nl[0], nl[1]):
        
        new[i_v].append(j_v)
        
    return new

def find_cycles(i_vert, cnl, max_length, cur_path, passed_edges):
    
    if len(cur_path)-1 == max_length:
        return []
    
    acc_cycles = []
    sort_cycles = []
    
    neighbs = cnl[i_vert]

    # if we are connected to something that is not the end
    # then we crossed multiple cycles
    for n in neighbs:
        edge = (np.min([i_vert, n]), np.max([i_vert, n]))
        if edge not in passed_edges:

            if n in cur_path[1:]:
                # path went too close to itself...
                return []

    # CHeck if we are at the end
    for n in neighbs:
        edge = (np.min([i_vert, n]), np.max([i_vert, n]))
        if edge not in passed_edges:
            
            if n == cur_path[0]:
                # found cycle
                return [cur_path]
    
    # Continue in all possible directions
    for n in neighbs:
        edge = (np.min([i_vert, n]), np.max([i_vert, n]))
        if edge not in passed_edges:

            cycs = find_cycles(n, cnl, max_length, cur_path + [n], passed_edges + [edge])
            for cyc in cycs:
                sorted_cyc = tuple(sorted(cyc))
                if sorted_cyc not in sort_cycles:
                    sort_cycles.append(sorted_cyc)
                    acc_cycles.append(cyc)
    
    return acc_cycles
    

def dumb_cycle_detection(ase_atoms_no_h, max_length):
    
    neighbor_list = ase.neighborlist.neighbor_list("ij", ase_atoms_no_h, 2.0)
    
    cycles = []
    sorted_cycles = []
    n_vert = np.max(neighbor_list)+1
    
    cnl = convert_neighbor_list(neighbor_list)
    
    for i_vert in range(n_vert):
        
        cycs = find_cycles(i_vert, cnl, max_length, [i_vert], [])
        for cyc in cycs:
            sorted_cyc = tuple(sorted(cyc))
            if sorted_cyc not in sorted_cycles:
                sorted_cycles.append(sorted_cyc)
                cycles.append(cyc)
    
    return cycles


def cycle_normal(cycle, h):
    cycle = np.array(cycle)
    centroid = np.mean(cycle, axis=0)

    points = cycle - centroid
    u, s, v = np.linalg.svd(points.T)
    normal = u[:, -1]
    normal /= np.linalg.norm(normal)
    if np.dot(normal, h*np.array([1, 1, 1])) < 0.0:
        normal *= -1.0
    return normal


def find_cycle_centers_and_normals(ase_atoms_no_h, cycles, h=0.0):
    """
    positive h means projection to z axis is positive and vice-versa
    """
    if h == 0.0:
        h = 1.0
    normals = []
    centers = []
    for cyc in cycles:
        cyc_p = []
        for i_at in cyc:
            cyc_p.append(ase_atoms_no_h[i_at].position)
        normals.append(cycle_normal(cyc_p, h))
        centers.append(np.mean(cyc_p, axis=0))
    return np.array(centers), np.array(normals)
