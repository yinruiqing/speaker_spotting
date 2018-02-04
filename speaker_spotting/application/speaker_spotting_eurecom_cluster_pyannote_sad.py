
import sys
sys.path.append("../")
import clustering
import numpy as np

from pyannote.audio.features import Precomputed
precomputed = Precomputed('/vol/work1/bredin/speaker_spotting/embeddings')


from pyannote.database import get_protocol, FileFinder
protocol = get_protocol('AMI.SpeakerSpotting.MixHeadset', progress=True)

from pyannote.core import Annotation,Segment, Timeline

# enrolment consists in summing all relevant embeddings
def speaker_spotting_enrol(current_enrolment):
    enrol_with = current_enrolment['enrol_with']
    embeddings = precomputed(current_enrolment)
    return np.sum(embeddings.crop(enrol_with), axis=0, keepdims=True)    

models = {}
for current_enrolment in protocol.development_enrolment():
    model_id = current_enrolment.pop('model_id')
    models[model_id] = speaker_spotting_enrol(current_enrolment)


from pyannote.parser import MDTMParser
cluster_mdtm = '/people/yin/projects/online_clustering/spotting/EURECOM-online-diarization-pyannote-VAD.dev.WithOffset.mdtm'
parser_dev = MDTMParser()
annotations_dev = parser_dev.read(cluster_mdtm)

REFERENCE = {}
for uri_part in annotations_dev.uris:
    uri = uri_part.split('_')[0] + '.Mix-Headset'
    if uri not in REFERENCE:
        REFERENCE[uri] = Annotation(uri=uri)
    REFERENCE[uri].update(annotations_dev(uri=uri_part, modality="speaker"))


# Trials

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.embedding.utils import cdist

# trial consists in comparing each embedding to the target embedding
def speaker_spotting_try_system2(current_trial):
    """ speaker spotting system based on the oracle 
    clustering system
    """
    # target model
    # record the model embedding vector 
    # and model id
    model = {}
    model_id = current_trial['model_id'] 
    model_embedding = models[current_trial['model_id']]
    model['mid'] = model_id
    model['embedding'] = model_embedding
    
    # where to look for this target
    try_with = current_trial['try_with']
    
    # precomputed embedding
    embeddings = precomputed(current_trial)
    
    # annotation of current file
    oracle_diarization = REFERENCE[current_trial['uri']].crop(current_trial['try_with'])

    
    # find index of first and last embedding fully included in 'try_with'
    indices = embeddings.sliding_window.crop(try_with, mode='strict')
    first, last = indices[0], indices[-1]

    onlineOracleClustering = clustering.OnlineOracleClustering(current_trial['uri'])
    start = embeddings.sliding_window[0].start
    data = np.zeros((len(embeddings.data), 1))
    for i, (window, _) in enumerate(embeddings):
        # make sure the current segment is in 'try_with'
        if i < first:
            start = window.end
            continue
        if i > last:
            break
            
        so_far = Segment(start, window.end)
        current_annotation = oracle_diarization.crop(so_far)
        score = 0.
        for segment, _, label in current_annotation.itertracks(label=True):
            example = {}
            example['label'] = label
            example['segment'] = segment
            example['embedding'] = embeddings.crop(segment, mode='center')
            example['indice'] = [i]
            # compute the distance with model
            example['distances'] = {}
            example['distances'][model['mid']] = list(cdist(example['embedding'], 
                                                            model['embedding'], 
                                                            metric='cosine').flatten())
            # update the online oracle clustering
            onlineOracleClustering.upadateCluster(example)
        if not onlineOracleClustering.empty():
            # compute the current score
            min_dist = min(onlineOracleClustering.modelClusterDistance(model))
            score = max(score, 2-min_dist)
        data[i] = score
        start = window.end
    
    # transform scores to sliding window features
    data = data[first:last+1]
    sliding_window = SlidingWindow(start=embeddings.sliding_window[first].start,
                                   duration=embeddings.sliding_window.duration,
                                   step=embeddings.sliding_window.step)
    
    return SlidingWindowFeature(data, sliding_window)



# Depending on the value of the detection threshold, the alarm will be triggered with a different latency.

def process_score(scores):
    min_score = 0
    res = []
    for (window, score) in scores:
        if score > min_score:
            res.append([window.end, score[0]])
            min_score = score[0]
    return res

def process_trial(trial, scores):
    res = {}
    pscores = process_score(scores)
    res['uri'] = trial['uri']
    res['model_id'] = trial['model_id']
    res['scores'] = pscores
    return res

llss = []

for current_trial in protocol.development_trial():
    reference = current_trial.pop('reference')
    hypothesis = speaker_spotting_try_system2(current_trial)
    llss.append(process_trial(current_trial, hypothesis))

import simplejson as json
with open('./results/ss_eurecom_cluster_pyannote_sad_representation.json', 'w') as outfile:  
    json.dump(llss, outfile)