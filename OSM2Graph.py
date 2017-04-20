# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:32:36 2017

@author: Shashwat
"""

import xml.etree.ElementTree


class Map(object):
    
    def __init__(self, map_file):
        self.root = xml.etree.ElementTree.parse(map_file).getroot()
        self.tags = set()
        for child in self.root.getchildren():
            self.tags.add(child.tag)
            
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
    def getNodeByTag(self, tags):
        '''
        Param: tags as a list of string
        Return: Node with a particular set of tags. For eg. nodes with tags of bus_stop.
        '''
        
        nodeList = self.getElementByTag('node', self.root)
        for node in nodeList:
            childTags = self.getElementByTag('tag', node)
            for childTag in childTags:
                pass
            

osmMap = Map('C:\\Users\\Shashwat\\Downloads\\map(2).osm')