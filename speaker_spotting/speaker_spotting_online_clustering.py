
# coding: utf-8

import clustering

from pyannote.audio.features import Precomputed
precomputed = Precomputed('/vol/work1/bredin/speaker_spotting/embeddings')
from pyannote.database import get_protocol, FileFinder
protocol = get_protocol('AMI.SpeakerSpotting.MixHeadset', progress=True)

def speaker_spotting_enrol(current_enrolment):
    enrol_with = current_enrolment['enrol_with']
    embeddings = precomputed(current_enrolment)
    return np.sum(embeddings.crop(enrol_with), axis=0, keepdims=True)    

models = {}
for current_enrolment in protocol.development_enrolment():
    model_id = current_enrolment.pop('model_id')
    models[model_id] = speaker_spotting_enrol(current_enrolment)

REFERENCE = {}
for current_file in protocol.development():
    uri = current_file['uri']
    if uri not in REFERENCE:
        REFERENCE[uri] = Annotation(uri=uri)
    REFERENCE[uri].update(current_file['annotation'])

# ## Trials
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.embedding.utils import cdist

from pyannote.core import Annotation,Segment, Timeline

# trial consists in comparing each embedding to the target embedding
def speaker_spotting_try_system4(current_trial):

    # target model
    model = {}
    model_id = current_trial['model_id']
    model_embedding = models[current_trial['model_id']]
    model['mid'] = model_id
    model['embedding'] = model_embedding
    # where to look for this target
    try_with = current_trial['try_with']
    
    # precomputed embedding
    embeddings = precomputed(current_trial)
    
    # find index of first and last embedding fully included in 'try_with'
    indices = embeddings.sliding_window.crop(try_with, mode='strict')
    speech_timeline = REFERENCE[current_trial['uri']].crop(current_trial['try_with']).get_timeline().support()
    indices_speech = embeddings.sliding_window.crop(speech_timeline, mode='strict')
    first, last = indices[0], indices[-1]
    onlineClustering = clustering.OnlineClustering('test', 
                                                   cdist(embeddings.data, 
                                                         embeddings.data, 
                                                         metric='cosine'))
    start = embeddings.sliding_window[0].start
    data = np.zeros((len(embeddings.data), 1))
    for i, (window, _) in enumerate(embeddings):
        if i < first or (i not in indices_speech):
            start = window.end
            continue
        if i > last:
            break
        so_far = Segment(start, window.end)
        score = 0.
        example = {}
        example['segment'] = so_far
        example['embedding'] = embeddings.crop(so_far, mode='center')
        example['indice'] = [i]
        example['distances'] = {}
        example['distances'][model['mid']] = list(cdist(example['embedding'], 
                                                        model['embedding'], 
                                                        metric='cosine').flatten())
        

        onlineClustering.upadateCluster(example)
        if not onlineClustering.empty():
            #min_dist = min(onlineClustering.computeDistances({'embedding': model}))
            min_dist = min(onlineClustering.modelDistance(model))
            score = max(score, 2-min_dist)
        data[i] = score
        start = window.end
    data = data[first:last+1]
    sliding_window = SlidingWindow(start=embeddings.sliding_window[first].start,
                                   duration=embeddings.sliding_window.duration,
                                   step=embeddings.sliding_window.step)
    
    return SlidingWindowFeature(data, sliding_window)



from pyannote.metrics.spotting import LowLatencySpeakerSpotting
metric = LowLatencySpeakerSpotting(thresholds=np.linspace(0, 2, 50))


for current_trial in protocol.development_trial():
    reference = current_trial.pop('reference')
    hypothesis = speaker_spotting_try_system4(current_trial)
    metric(reference, hypothesis)
