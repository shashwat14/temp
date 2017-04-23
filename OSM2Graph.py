# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:32:36 2017

@author: Shashwat
"""

import xml.etree.ElementTree
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Map(object):
    
    def __init__(self, map_file):
        self.root = xml.etree.ElementTree.parse(map_file).getroot()
        self.tags = set()
        children = self.root.getchildren()
        for child in children:
            self.tags.add(child.tag)
        boundElement = children[0]
        self.minlat = float(boundElement.get('minlat'))
        self.maxlat = float(boundElement.get('maxlat'))
        self.minlon = float(boundElement.get('minlon'))
        self.maxlon = float(boundElement.get('maxlon'))
        
    def getTags(self):
        '''
        Param: None
        Return: set of available tags for given map
        '''
        return self.tags
    
    def getElementByTag(self, tag, element=None):
        '''
        Param: tag as string. Options available - way, node, bounds, member
        Return: list of element with given tag
        '''
        if element is not None:
            return element.findall(tag)
        return self.root.findall(tag)
    
    def getNodeTags(self):
        '''
        Param: None
        Return: Set of available tags within a node
        '''
        tagSetKey = set()
        tagSetValue = set()
        nodeList = self.getElementByTag('node', self.root)
        for node in nodeList:
            childTags = self.getElementByTag('tag', node)
            for childTag in childTags:
                tagSetKey.add(childTag.get('k'))
                tagSetValue.add(childTag.get('v'))
        return tagSetKey, tagSetValue
            
    def getWayTags(self):
        '''
        Param: None
        Return: Set of available tags within a way
        '''
        tagKeySet = set()
        tagValueSet = set()
        wayList = self.getElementByTag('way', self.root)
        for way in wayList:
            childTags = self.getElementByTag('tag', way)
            for childTag in childTags:
                tagKeySet.add(childTag.get('k'))
                tagValueSet.add(childTag.get('v'))
        return tagKeySet, tagValueSet
    
    #DO THIS
    def getNodeByTag(self, tags=None):
        '''
        Param: tags as a list of string
        Return: List of nodes with a particular set of tags. For eg. nodes with tags of bus_stop.
        '''
        
        nodeObjectList = []
        nodeList = self.getElementByTag('node', self.root)
        
        for node in nodeList:
            ref = node.get('id')
            lat = node.get('lat')
            lon = node.get('lon')
            
            childTags = self.getElementByTag('tag', node)
            key_value_pair = {}
            for childTag in childTags:
                key = childTag.get('k')
                value = childTag.get('v')
                key_value_pair[key] = value
            
            nodeObject = Node(ref, lat, lon, key_value_pair)
            nodeObjectList.append(nodeObject)
        return nodeObjectList
    
    def getWayByTag(self, tags=None):
        '''
        Param: tags as a list of string
        Return: List of ways with a particular set of tags. For eg. ways with tags of highway
        '''
        
        wayList = self.getElementByTag('way', self.root)
        wayObjectList = []
        for way in wayList:
            
            ref = way.get('id')
            #Save list of nodes making up a way
            childTags = self.getElementByTag('nd', way)
            lst = []
            for childTag in childTags:
                if childTag.get('ref') is not None:
                    lst.append(childTag.get('ref'))
                    
            #Save list of key value pair for each way
            childTags = self.getElementByTag('tag', way)
            key_value_pairs = {}
            for childTag in childTags:
                key = childTag.get('k')
                value = childTag.get('v')
                key_value_pairs[key] = value
            
            #Create Way Object
            wayObject = Way(ref, lst, key_value_pairs)
            wayObjectList.append(wayObject)
            
        return wayObjectList
    
    def getNodeHash(self):
        '''
        Param: None
        Return: Dictionary of node references to node objects
        '''
        
        nodeList = self.getNodeByTag()
        nodeHash = {}
        for node in nodeList:
            nodeHash[node.ref] = node
        return nodeHash
    
    def generateAdjacencyList(self, wayList):
        '''
        Param: list of ways where each way is Way object
        Return: Dictionary as adjacency list
        '''
        
        adjacencyList = {}
        for way in wayList:
            if 'highway' not in way.dict.keys() :#or way.dict['highway'] != 'footway':
                continue
            nodes_refs = way.nodes
            for i in range(1, len(nodes_refs)):
                node_i = nodes_refs[i-1]
                node_j = nodes_refs[i]
                nodeSet = set(adjacencyList.keys())
                if node_i not in nodeSet and node_j not in nodeSet:
                    adjacencyList[node_i] = [node_j]
                    adjacencyList[node_j] = [node_i]
                elif node_i not in nodeSet and node_j in nodeSet:
                    adjacencyList[node_i] = [node_j]
                    adjacencyList[node_j].append(node_i)
                elif node_i in nodeSet and node_j not in nodeSet:
                    adjacencyList[node_i].append(node_j)
                    adjacencyList[node_j] = [node_i]
                else:
                    adjacencyList[node_i].append(node_j)
                    adjacencyList[node_j].append(node_i)
        return adjacencyList
                    
    def toXY(self, lat, lon):
        '''
        Custom function for user defined transformation to x,y coordinates
        Param: list of latitude in degrees, list of longitudes in degrees
        Return: List of X and Y
        '''
        
        lat-=1.3
        lon-=103.77
        lat*=100000*2
        lon*=100000*2
        
        return lat, lon
        
class Way(object):
    
    def __init__(self, ref, nd, key_value_pairs):
        self.ref = ref
        self.nodes = nd
        self.dict = key_value_pairs

class Node(object):
    
    def __init__(self, ref, lat, lon, key_value_pairs):
        self.ref = ref
        self.lat = float(lat)
        self.lon = float(lon)
        self.dict = key_value_pairs

map = Map('C:\\Users\\Shashwat\\Downloads\\map(2).osm')
nodeHash = map.getNodeHash()
wayList = map.getWayByTag()
al = map.generateAdjacencyList(wayList)
lat = []
lon = []
for each in al.keys():
    latitude = float(nodeHash[each].lat)
    longitude = float(nodeHash[each].lon)
    if latitude > map.minlat and latitude < map.maxlat and longitude < map.maxlon and longitude > map.minlon:
        lat.append(latitude)
        lon.append(longitude)
lat = np.array(lat)
lon = np.array(lon)
lat, lon = map.toXY(lat, lon)


#rows = latitude
#cols = longitude
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
img = np.zeros((900*2, 600*2))
for l1, l2 in zip(lon, lat):
    cv2.circle(img, (int(l1), 2000-int(l2)), 2, 255, -1)

for init in al.keys():
    init_node = nodeHash[init]
    ilat, ilon = map.toXY(np.array([init_node.lat]), np.array([init_node.lon]))
    fins = al[init]
    for each in fins:
        each_node = nodeHash[each]
        flat, flon = map.toXY(np.array([each_node.lat]), np.array([each_node.lon]))
        cv2.line(img, (int(ilon), 2000-int(ilat)),(int(flon), 2000-int(flat)), 255, 1)
cv2.imshow("Frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("C:\\Users\\Shashwat\\Desktop\\osm_utown_map.png", img)