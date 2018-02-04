# coding: utf-8
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
for current_enrolment in protocol.test_enrolment():
    model_id = current_enrolment.pop('model_id')
    models[model_id] = speaker_spotting_enrol(current_enrolment)

REFERENCE = {}
for current_file in protocol.test():
    uri = current_file['uri']
    if uri not in REFERENCE:
        REFERENCE[uri] = Annotation(uri=uri)
    REFERENCE[uri].update(current_file['annotation'])


from pyannote.parser import MDTMParser
sad_dev = '/people/yin/projects/online_clustering/spotting_test/AMI.SpeakerSpotting.MixHeadset.test.mdtm'
parser_dev = MDTMParser()
annotations_dev = parser_dev.read(sad_dev)
SAD = {}
for item in protocol.test():
    uri = item['uri']
    SAD[uri] = annotations_dev(uri=uri, modality="speaker").get_timeline().support()


# Trials

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.embedding.utils import cdist

# trial consists in comparing each embedding to the target embedding
def speaker_spotting_try(current_trial):

    # target model
    model = models[current_trial['model_id']]
    # where to look for this target
    try_with = current_trial['try_with']
    
    # precomputed embedding
    embeddings = precomputed(current_trial)
    
    # find index of first and last embedding fully included in 'try_with'
    indices = embeddings.sliding_window.crop(try_with, mode='strict')
    first, last = indices[0], indices[-1]
    
    speech_timeline = SAD[current_trial['uri']]
    indices_speech = embeddings.sliding_window.crop(speech_timeline, mode='strict')

    # compare all embeddings to target model
    scores = 2. - cdist(embeddings.data, model, metric='cosine')

    data = np.zeros((len(embeddings.data), 1))
    for i, (window, _) in enumerate(embeddings):
        # make sure the current segment is in 'try_with'
        if i < first or (i not in indices_speech):
            continue
        if i > last:
            break
        data[i] = scores[i]

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

for current_trial in protocol.test_trial():
    reference = current_trial.pop('reference')
    hypothesis = speaker_spotting_try(current_trial)
    llss.append(process_trial(current_trial, hypothesis))

import simplejson as json
with open('./results/ss_no_cluster_pyannote_sad_with_overlap_test.json', 'w') as outfile:  
    json.dump(llss, outfile)
