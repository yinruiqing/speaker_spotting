import numpy as np
from pyannote.core import Annotation,Segment, Timeline
from pyannote.audio.embedding.utils import cdist
import itertools
from itertools import combinations


class Cluster():
    def __init__(self, label, dist_metric=cdist):
        """ Cluster class 
        It is used to segments clustering. 

        Parameters
        ----------
        label: string
            name of cluster
        dist_metric: optional
            distance metric to compute distance
            between clusters and models 

        """
        self.label = label       # cluster name
        self.representation = 0  # cluster embedding
        self.dist_metric = cdist # distance metric
        self.embeddings = []     # embedding of segments in this cluster
        self.indices = []
        self.segments = []       # segements in this cluster
        self.distances = {}

    def distance(self,data):
        """ 
        compute the distance between cluster 
        and segment

        Parameters
        ----------
        data: dict
            example:
                { 'embedding': np.array(...),
                  'segment': Segment(10, 20) }
        """
        feature = np.sum(data['embedding'], axis=0, keepdims=True)

        dists = []
        for embedding in self.embeddings:
            dists += list(self.dist_metric(embedding, feature, metric='cosine').flatten())
        return np.mean(dists)
        
    def distance2(self,data):
        """ 
        compute the distance between cluster 
        and segment

        Parameters
        ----------
        data: dict
            example:
                { 'embedding': np.array(...),
                  'segment': Segment(10, 20) }
        """
        feature = np.sum(data['embedding'], axis=0, keepdims=True)
        return self.dist_metric(self.representation, feature, metric='cosine')[0, 0]

    def distanceModel(self, model):
        """ 
        compute the distance between cluster 
        and model

        Parameters
        ----------
        model: dict
            example:
                { 'mid': str,
                  'embedding': np.array }
        """
        # if distance between model and embedding are stored
        # in cluster.distance, than return the average of distance

        if model['mid'] in self.distances:
            return np.mean(self.distances[model['mid']])

        embeddings = np.concatenate(self.embeddings, axis=0)
        dists = []
        for embedding in self.embeddings:
            dists += list(self.dist_metric(embedding, model['embedding'], metric='cosine').flatten())
        return np.mean(dists)

    def distanceModelCluster(self, model):
        """ 
        compute the distance between cluster embedding
        and model

        Parameters
        ----------
        model: dict
            example:
                { 'mid': str,
                  'embedding': np.array }
        """

        return self.dist_metric(self.representation, model['embedding'], metric='cosine')[0, 0]


    # def distanceModel(self,model):
    #     """
    #     compute distance between cluster and model
    #     """
    #     return self.dist_metric(self.representation, model, metric='cosine')[0, 0]
    
    def updateCluster(self,data):
        """
        update the cluster:
            add new segment embedding
            update the cluster embedding
            add segment to cluster 
            add distances between model and cluster
        """
        self.embeddings.append(data['embedding'])
        self.representation += np.sum(data['embedding'], axis=0, keepdims=True)
        self.segments.append(data['segment'])
        self.indices += (data['indice'])
        if 'distances' in data:
            for mid in data['distances']:
                if mid not in self.distances:
                    self.distances[mid] = []
                self.distances[mid] += data['distances'][mid]
        return
    
    def mergeClusters(self,cluster):
        """
        update the cluster by another cluster
        """
        self.embeddings.update(cluster.embeddings)
        self.segments.update(cluster.segments)
        return


class OnlineClustering():
    """ online clustering class
    compare new comming segment with clusters, then decide 
    create a new cluster, or add it to a existing cluster. 
    When the distance between the new coming segment and 
    clusters is larger than a predetermined threshold
    then it will be added to the closest cluster. Otherwise,
    add a new cluster

    Parameters
    ----------
    uri: name
    threshold: float, optional
        distance threhold, when the distance exceding 
        the threshold, a new cluster created
    generator_method: str, optional 
        generator of names
        value should be "string" or "int"

    """
    def __init__(self, uri, distance_matrix,
                 threshold=0.5, 
                generator_method='string'):
        self.uri = uri              # cluster name
        self.threshold = threshold  # threshold to decide add a new cluster or update a existing cluster
        self.distance_matrix = distance_matrix # distance between embeddings 
        
        self.clusters = []          #store the current clusters
        self.generator_method = generator_method # generate cluster names
        
        if self.generator_method == 'string':
            from pyannote.core.util import string_generator
            self.generator = string_generator()
        elif self.generator_method == 'int':
            from pyannote.core.util import int_generator
            self.generator = int_generator()

    
    def getLabels(self):
        """
        returns all the cluster labels
        """
        return [cluster.label for cluster in self.clusters]
    
    def getAnnotations(self):
        """
        return annotations of cluster result
        todo: add warning when clusters is empty
        """
        annotation = Annotation(uri=self.uri, modality='speaker')
        for cluster in self.clusters:
            for seg in cluster.segments:
                annotation[seg] = cluster.label
        
        return annotation
    
    def addCluster(self,data):
        """
        create a new cluster
        """
        label = next(self.generator)
        cluster = Cluster(label)
        cluster.updateCluster(data)
        self.clusters.append(cluster)
        return
        
    def computeDistances(self, data):
        """Compare new coming data with clusters"""
        # return [cluster.distance(data) for cluster in self.clusters]
        distances = []
        for cluster in self.clusters:
            i = cluster.indices
            j = data['indice']
            indexs = list(itertools.product(i,j))
            distances.append(np.mean([self.distance_matrix[i] for i in indexs]))
        return distances

    def computeDistances2(self, data):
        """Compare new coming data with clusters"""
        # return [cluster.distance(data) for cluster in self.clusters]
        
        return [cluster.distance2(data) for cluster in self.clusters]
        
    
    def upadateCluster(self,data):
        """add new coming data to clustering result
        If the distance between data and clusters are smaller 
        than the threshold, add the data to the 
        closest cluster. Otherwise, create a new cluster
        """
        if len(self.clusters) == 0:
            self.addCluster(data)
            return
        
        distances = self.computeDistances(data)
        if min(distances) > self.threshold:
            self.addCluster(data)
        else:
            indice = distances.index(min(distances))
            to_update_cluster = self.clusters[indice]
            to_update_cluster.updateCluster(data)
        return

    def upadateCluster2(self,data):
        """add new coming data to clustering result
        If the distance between data and clusters are smaller 
        than the threshold, add the data to the 
        closest cluster. Otherwise, create a new cluster
        """
        if len(self.clusters) == 0:
            self.addCluster(data)
            return
        
        distances = self.computeDistances2(data)
        if min(distances) > self.threshold:
            self.addCluster(data)
        else:
            indice = distances.index(min(distances))
            to_update_cluster = self.clusters[indice]
            to_update_cluster.updateCluster(data)
        return
    
    def modelDistance(self, model):
        """Compare model with clusters

        Returns:
        --------
        distances: list of float

        """
        distances = []
        for cluster in self.clusters:
            distances.append(cluster.distanceModel(model))
        return distances

    def modelClusterDistance(self, model):
        """Compare model with clusters

        Returns:
        --------
        distances: list of float

        """
        distances = []
        for cluster in self.clusters:
            distances.append(cluster.distanceModelCluster(model))
        return distances

    def empty(self):
        if len(self.clusters)==0:
            return True
        return False



class OnlineOracleClustering():
    """ online oracle clustering 
    All segments have been clustered
    It will compute the cluster embedding online

    Parameters
    ----------
    uri: name

    """

    def __init__(self, uri):

        self.uri = uri
        self.clusters = {}
    
    def getLabels(self):
        """
        return all the cluster labels

        Returns:
        --------
        labels: list of str

        """
        return self.clusters.keys()
    
    def getAnnotations(self):
        """
        annotations of cluster result
        Returns:
        --------
        annotation: pyannote.core.Annotation

        todo: 
        add warning when clusters is empty
        """
        annotation = Annotation(uri=self.uri, modality='speaker')
        for cluster_id in self.clusters:
            for seg in self.clusters[cluster_id].segments:
                annotation[seg] = self.clusters[cluster_id].label
        
        return annotation
    
    def addCluster(self,data):
        """
        create a new cluster
        """
        label = data['label']
        cluster = Cluster(label)
        cluster.updateCluster(data)
        self.clusters[label] = cluster
        return
        
    # def computeDistances(self, data):
    #     """Compare new coming data with clusters

    #     Returns:
    #     --------
    #     distances: list of float

    #     """
    #     distances = []
    #     for label in self.clusters:
    #         distances.append(self.clusters[label].distance(data))
    #     return distances

    def modelDistance(self, model):
        """Compare model with clusters

        Returns:
        --------
        distances: list of float

        """
        distances = []
        for label in self.clusters:
            distances.append(self.clusters[label].distanceModel(model))
        return distances


    def modelClusterDistance(self, model):
        """Compare model with clusters

        Returns:
        --------
        distances: list of float

        """
        distances = []
        for label in self.clusters:
            distances.append(self.clusters[label].distanceModelCluster(model))
        return distances

    def allModelClusterDistance(self, model):
        """Compare model with clusters

        Returns:
        --------
        distances: list of float

        """
        distances = {}
        for label in self.clusters:
            distances[label] = self.clusters[label].distanceModelCluster(model)
        return distances
    
    def modelsDistances(self, models):
        """Compare all models with clusters

        Returns:
        --------
        distances: dict
            key: cluster label
            value: min distance with clusters

        """
        distances = {}
        for label, model in models.items():
            distances[label] = min(self.modelDistance(model))

        return distances

    
    def upadateCluster(self,data):
        """add new coming data to clustering result
        If the cluster is existed, add the data to the 
        corresponding cluster. Otherwise, create a new cluster
        """
        if data['label'] in self.clusters:
            self.clusters[data['label']].updateCluster(data)
        else:
            self.addCluster(data)
        return
    
    def empty(self):
        if len(self.clusters)==0:
            return True
        return False
    
class HierarchicalClustering():
    def __init__(self, uri, stop_threshold=0.5,
                generator_method='int',
                dist_metric=cdist):
        """ Hierarchical Clustering
        Each observation starts in its own cluster, and pairs of clusters 
        with minimum distance are merged as one moves up the hierarchy.
        This process stop when the minimum distance larger than a predetemined 
        threshold 

        Parameters
        ----------
        uri: string
            name of cluster
        stop_threshold: float
            stop the clustering when the min distance is 
            larger than stop_threshold
        generator_method: str, optional 
            generator of names
            value should be "string" or "int"
        dist_metric: optional
            distance metric to compute distance
            between clusters and models 

        """
        self.uri = uri
        self.clusters = []
        self.tree = []
        self.generator_method = generator_method
        self.stop_threshold = stop_threshold
        self.dist_metric = cdist
        #self.annotations = Annotation(uri=self.uri)

        if self.generator_method == 'string':
            from pyannote.core.util import string_generator
            self.generator = string_generator()
        elif self.generator_method == 'int':
            from pyannote.core.util import int_generator
            self.generator = int_generator()

    
    def getLabels(self):
        """
        return all the cluster labels

        Returns:
        --------
        labels: list of str

        """
        return [cluster.label for cluster in self.clusters]
    
    def getAnnotations(self):
        """
        annotations of cluster result
        Returns:
        --------
        annotation: pyannote.core.Annotation

        todo: 
        add warning when clusters is empty
        """
        annotation = Annotation(uri=self.uri, modality='speaker')
        for cluster in self.clusters:
            for seg in cluster.segments:
                annotation[seg] = cluster.label
        
        return annotation
    
    def addCluster(self,data):
        """ add new cluster """
        label = next(self.generator)
        cluster = Cluster(label)
        cluster.updateCluster(data)
        self.clusters.append(cluster)
        return
        
    def computeDistances(self, data):
        """Compare new coming data with clusters"""
        return [cluster.distance(data) for cluster in self.clusters]
        

    def mergeClusters(self, cluster1, cluster2):
        """ merge two clusters 

        Parameter:
        --------
        cluster1: clustering.Cluster
        cluster2: clustering.Cluster

        Returns:
        --------
        cluster: clustering.Cluster
        """
        label = next(self.generator)
        cluster = Cluster(label)
        cluster.representation = cluster1.representation + cluster2.representation
        cluster.embeddings = cluster1.embeddings + cluster2.embeddings
        cluster.segments = cluster1.segments + cluster2.segments
        return cluster

    def distance(self, cluster1, cluster2):
        """ compute distance between two clusters

        Parameter:
        --------
        cluster1: clustering.Cluster
        cluster2: clustering.Cluster

        Returns:
        --------
        distance: float     

        """
        return self.dist_metric(cluster1.representation, 
            cluster2.representation, metric='cosine')

    def fit(self, X):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X: tuple (ebmedding, segment) list
            embedding: array-like
            segment: pyannote.core.Segment

        """
        for embedding, segment in X:
            data = {}
            data['embedding'] = embedding
            data['segment'] = segment
            self.addCluster(data)
        for cluster in self.clusters:
            self.tree.append((-1,-1,cluster.label,0))
        clustersOut = []
        min_distance = 0
        while len(self.clusters) > 1 and min_distance < self.stop_threshold:
            # calculating distances
            distances = [(self.distance(i, j),(i,j)) for i, j in combinations(self.clusters,2)]
            # merge closest clusters
            min_distance = min(distances)[0]
            i,j = min(distances)[1]
            self.clusters.remove(i)
            self.clusters.remove(j)
            new_cluster = self.mergeClusters(i,j)
            self.clusters.append(new_cluster)
            self.tree.append((i.label,j.label,new_cluster.label,min_distance))
            self.min_dist = min_distance
        return