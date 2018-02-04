
# coding: utf-8

# ```bash
# $ pip install pyannote.metrics==1.4.1
# $ pip install pyannote.db.odessa.ami==0.5.1
# ```

# In[26]:

get_ipython().magic('pylab inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[27]:

sys.path.append("../../projects/online_clustering/")
import clustering


# In[28]:

from pyannote.audio.features import Precomputed
precomputed = Precomputed('/vol/work1/bredin/speaker_spotting/embeddings')


# In[29]:

from pyannote.database import get_protocol, FileFinder
protocol = get_protocol('AMI.SpeakerSpotting.MixHeadset', progress=True)


# ## Enrolment

# In[30]:

# enrolment consists in summing all relevant embeddings
def speaker_spotting_enrol(current_enrolment):
    enrol_with = current_enrolment['enrol_with']
    embeddings = precomputed(current_enrolment)
    return np.sum(embeddings.crop(enrol_with), axis=0, keepdims=True)    


# In[31]:

models = {}
for current_enrolment in protocol.test_enrolment():
    model_id = current_enrolment.pop('model_id')
    models[model_id] = speaker_spotting_enrol(current_enrolment)


# In[32]:

REFERENCE = {current_file['uri']: current_file['annotation'] for current_file in protocol.test()}


# ## Trials

# In[46]:

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.embedding.utils import cdist
from pyannote.core import Annotation,Segment, Timeline
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
            min_dist = min(onlineOracleClustering.modelDistance(model))
            score = max(score, 2-min_dist)
        data[i] = score
        start = window.end
    import pdb
    #pdb.set_trace()
    
    # transform scores to sliding window features
    data = data[first:last+1]
    sliding_window = SlidingWindow(start=embeddings.sliding_window[first].start,
                                   duration=embeddings.sliding_window.duration,
                                   step=embeddings.sliding_window.step)
    
    return SlidingWindowFeature(data, sliding_window)


# ### Example

# In the figure below, 
# - the top timeline shows the region where target speaker is actually speaking.
# - the bottom curve shows the actual output score of the system over time

# In[11]:

from pyannote.core.notebook import notebook
from pyannote.core import Segment
trials = list(get_protocol('AMI.SpeakerSpotting.MixHeadset').test_trial())
trial = trials[1050]
notebook.crop = Segment(300, 600)
figsize(15, 4)
plt.subplot(211)
notebook.plot_timeline(trial['reference'].support(), time=False)
plt.subplot(212)
hypothesis = speaker_spotting_try_system2(trial)
notebook.plot_feature(hypothesis)
plot(notebook.crop, [1.75, 1.75]);
legend(['Detection score', 'Detection threshold']);


# Depending on the value of the detection threshold, the alarm will be triggered with a different latency.

# Let's run all trials...

# In[48]:

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


# In[15]:

from pyannote.metrics.spotting import LowLatencySpeakerSpotting
metric = LowLatencySpeakerSpotting(thresholds=np.linspace(0, 2, 50))
llss = []
for current_trial in protocol.test_trial():
    reference = current_trial.pop('reference')
    hypothesis = speaker_spotting_try_system2(current_trial)
    llss.append(process_trial(current_trial, hypothesis))
    metric(reference, hypothesis)


# In[ ]:

import simplejson as json
with open('llss.txt', 'w') as outfile:  
    json.dump(llss, outfile)


# ## Results

# In[16]:

thresholds, fpr, fnr, eer, cdet  = metric.det_curve(cost_miss=1000, cost_fa=1, prior_target=0.001)


# ### DET curve

# In[18]:

figsize(5, 5)
loglog(fpr, fnr); xlabel('False alarm rate'); ylabel('False rejection rate'); xlim(0, 0.5); ylim(0, 0.5);
plot([eer], [eer], 'o')
xlim(1e-2, 0.9); ylim(1e-2, 0.9);
title('EER = {eer:.2f}%'.format(eer=100 * eer));


# ### $C_{det}$ cost as a function of the detection threshold

# In[19]:

plot(thresholds, cdet); xlim(1., 2.); xlabel('Detection threshold'); ylabel('$C_{det}$')


# ### Latency as a function of the detection threshold

# In[21]:

thresholds, fpr, fnr, eer, cdet, speaker_latency, absolute_latency = metric.det_curve(
    cost_miss=1000, cost_fa=1, prior_target=0.001, return_latency=True)


# In[22]:

plot(thresholds, speaker_latency, label='Speaker latency')
plot(thresholds, absolute_latency, label='Absolute latency')
legend(); ylim(0, 200)
xlabel('Detection threshold'); ylabel('Latency (s)');


# ### $C_{det}$ as a function of the latency

# In[23]:

semilogx(speaker_latency, cdet)
xlabel('Speaker latency (s)'); ylabel('Cdet');


# In[39]:

hypothesis


# In[37]:

reference 


# In[44]:

metric = LowLatencySpeakerSpotting(thresholds=np.linspace(0, 2, 50))
for current_trial in protocol.test_trial():
    reference = current_trial.pop('reference')
    hypothesis = speaker_spotting_try_system2(current_trial)
    metric(reference, hypothesis)
    break


# In[45]:

print(REFERENCE[current_trial['uri']].crop(current_trial['try_with']))

