"""
Speaker-spotting
Usage:
  speaker-spotting.py oracle [--subset=<subset>] <database.task.protocol> <output_file>
  speaker-spotting.py automatic [--subset=<subset>] <database.task.protocol> <diarization.mdtm> <output_file>
  speaker-spotting.py segment [--subset=<subset> --automatic-sad --sad=<sad.mdtm>] <database.task.protocol> <output_file>
  speaker-spotting.py -h | --help
  speaker-spotting.py --version
Options:
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  <diarization.mdtm>         Diarization results 
  --subset=<subset>          Evaluated subset (train|developement|test) [default: development]
  <output_file>              File path to store the score
  oracle                     Use oracle diarization result
  automatic                  Use automatic diarization result
  segment                    Compute the score directly
  -h --help                  Show this screen.
"""

from docopt import docopt
import sys
sys.path.append("../")
import clustering
import numpy as np


from pyannote.core import Annotation,Segment, Timeline
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.embedding.utils import cdist
from pyannote.database import get_protocol, FileFinder

# enrolment consists in summing all relevant embeddings
def speaker_spotting_enrol(current_enrolment):
    enrol_with = current_enrolment['enrol_with']
    embeddings = precomputed(current_enrolment)
    return np.sum(embeddings.crop(enrol_with), axis=0, keepdims=True)    

def speaker_spotting_try_diarization(current_trial):
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

    # start online oracle diarization
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
        
        # segment to be processed 
        so_far = Segment(start, window.end)
        current_annotation = oracle_diarization.crop(so_far)
        score = 0.
        for segment, _, label in current_annotation.itertracks(label=True):

            example = {}
            example['label'] = label
            example['segment'] = segment
            example['embedding'] = embeddings.crop(segment, mode='center')
            example['indice'] = [i]
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


def speaker_spotting_try_segment(current_trial):

    # target model
    model = models[current_trial['model_id']]
    # where to look for this target
    try_with = current_trial['try_with']

    # precomputed embedding
    embeddings = precomputed(current_trial)

    # find index of first and last embedding fully included in 'try_with'
    indices = embeddings.sliding_window.crop(try_with, mode='strict')
    first, last = indices[0], indices[-1]

    speech_timeline = SPEECH[current_trial['uri']].crop(current_trial['try_with']).get_timeline().support()
    indices_speech = embeddings.sliding_window.crop(speech_timeline, mode='strict')

    # compare all embeddings to target model
    scores = 2. - cdist(embeddings.data, model, metric='cosine')

    data = np.zeros((len(embeddings.data), 1))
    for i, (window, _) in enumerate(embeddings):
        # make sure the current segment is in 'try_with' and speech part
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


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Speaker-spotting')
    # protocol
    protocol_name = arguments['<database.task.protocol>']
    protocol = get_protocol(protocol_name, progress=True)

    # subset (train, development, or test)
    subset = arguments['--subset']
    output_file = arguments['<output_file>']
    from pyannote.audio.features import Precomputed
    precomputed = Precomputed('/vol/work1/bredin/speaker_spotting/embeddings')

    models = {}
    enrolments = getattr(protocol, '{subset}_enrolment'.format(subset=subset))()
    for current_enrolment in enrolments:
        model_id = current_enrolment.pop('model_id')
        models[model_id] = speaker_spotting_enrol(current_enrolment)
    if arguments['oracle']:
        REFERENCE = {}
        for current_file in getattr(protocol,subset)():
            uri = current_file['uri']
            if uri not in REFERENCE:
                REFERENCE[uri] = Annotation(uri=uri)
            REFERENCE[uri].update(current_file['annotation'])

        llss = []
        trials = getattr(protocol, '{subset}_trial'.format(subset=subset))()
        for current_trial in trials:
            reference = current_trial.pop('reference')
            hypothesis = speaker_spotting_try_diarization(current_trial)
            llss.append(process_trial(current_trial, hypothesis))

        # store the result to a file
        import simplejson as json
        with open(output_file, 'w') as outfile:  
            json.dump(llss, outfile)

    if arguments['automatic']:
        from pyannote.parser import MDTMParser
        diarization_mdtm = arguments['<diarization.mdtm>']
        parser = MDTMParser()
        annotations = parser.read(diarization_mdtm)

        REFERENCE = {}
        for uri_part in annotations.uris:
            uri = uri_part.split('_')[0] + '.Mix-Headset'
            if uri not in REFERENCE:
                REFERENCE[uri] = Annotation(uri=uri)
            REFERENCE[uri].update(annotations(uri=uri_part, modality="speaker"))

        llss = []
        trials = getattr(protocol, '{subset}_trial'.format(subset=subset))()

        for current_trial in trials:
            reference = current_trial.pop('reference')
            hypothesis = speaker_spotting_try_diarization(current_trial)
            llss.append(process_trial(current_trial, hypothesis))

        import simplejson as json
        with open(output_file, 'w') as outfile:  
            json.dump(llss, outfile)


    if arguments['segment']:

        if arguments['--automatic-sad']:
            from pyannote.parser import MDTMParser
            sad_mdtm = arguments['--sad']
            parser = MDTMParser()
            annotations = parser.read(sad_mdtm)
            SPEECH = {}
            for item in getattr(protocol,subset)():
                uri = item['uri']
                SPEECH[uri] = annotations(uri=uri, modality="speaker")
        else:
            SPEECH = {}
            for current_file in getattr(protocol,subset)():
                uri = current_file['uri']
                if uri not in SPEECH:
                    SPEECH[uri] = Annotation(uri=uri)
                SPEECH[uri].update(current_file['annotation'])


        llss = []
        trials = getattr(protocol, '{subset}_trial'.format(subset=subset))()
        for current_trial in trials:
            reference = current_trial.pop('reference')
            hypothesis = speaker_spotting_try_segment(current_trial)
            llss.append(process_trial(current_trial, hypothesis))

        import simplejson as json
        with open(output_file, 'w') as outfile:  
            json.dump(llss, outfile)
