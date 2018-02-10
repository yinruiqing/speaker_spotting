#python speaker-spotting.py oracle /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset  result/1s/AMI.SpeakerSpotting.MixHeadset.oracleDiarization.development.embedding.json

#python speaker-spotting.py oracle /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset  result/3s/AMI.SpeakerSpotting.MixHeadset.oracleDiarization.development.embedding.json

#python speaker-spotting.py automatic /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset /people/yin/projects/speaker_spotting/speaker_spotting/AMI_dev_th_0.60.0.256.100.mdtm result/1s/AMI.SpeakerSpotting.MixHeadset.automaticDiarization.development.embedding.json

#python speaker-spotting.py automatic /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset /people/yin/projects/speaker_spotting/speaker_spotting/AMI_dev_th_0.60.0.256.100.mdtm result/3s/AMI.SpeakerSpotting.MixHeadset.automaticDiarization.development.embedding.json

python speaker-spotting.py segment /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset  result/1s/AMI.SpeakerSpotting.MixHeadset.segmentOSAD.development.embedding.json

python speaker-spotting.py segment /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset  result/3s/AMI.SpeakerSpotting.MixHeadset.segmentOSAD.development.embedding.json

#python speaker-spotting.py segment --automatic-sad --sad=/people/yin/projects/online_clustering/spotting/AMI.SpeakerSpotting.MixHeadset.development.mdtm /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset  result/1s/AMI.SpeakerSpotting.MixHeadset.segmentASAD.development.embedding.json

#python speaker-spotting.py segment --automatic-sad --sad=/people/yin/projects/online_clustering/spotting/AMI.SpeakerSpotting.MixHeadset.development.mdtm /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset  result/3s/AMI.SpeakerSpotting.MixHeadset.segmentASAD.development.embedding.json


#python speaker-spotting.py oracle --subset=test /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset  result/1s/AMI.SpeakerSpotting.MixHeadset.oracleDiarization.test.embedding.json

#python speaker-spotting.py oracle --subset=test /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset  result/3s/AMI.SpeakerSpotting.MixHeadset.oracleDiarization.test.embedding.json

#python speaker-spotting.py automatic --subset=test /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset /people/yin/projects/speaker_spotting/speaker_spotting/AMI_tst_th_0.60.0.256.100.mdtm result/1s/AMI.SpeakerSpotting.MixHeadset.automaticDiarization.test.embedding.json

#python speaker-spotting.py automatic --subset=test /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset /people/yin/projects/speaker_spotting/speaker_spotting/AMI_tst_th_0.60.0.256.100.mdtm result/3s/AMI.SpeakerSpotting.MixHeadset.automaticDiarization.test.embedding.json

python speaker-spotting.py segment --subset=test /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset  result/1s/AMI.SpeakerSpotting.MixHeadset.segmentOSAD.test.embedding.json

python speaker-spotting.py segment --subset=test /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset  result/3s/AMI.SpeakerSpotting.MixHeadset.segmentOSAD.test.embedding.json

#python speaker-spotting.py segment --subset=test --automatic-sad --sad=/people/yin/projects/online_clustering/spotting_test/AMI.SpeakerSpotting.MixHeadset.test.mdtm /vol/work1/bredin/for_ruiqing/odyssey_llss/1s/ AMI.SpeakerSpotting.MixHeadset  result/1s/AMI.SpeakerSpotting.MixHeadset.segmentASAD.test.embedding.json

#python speaker-spotting.py segment --subset=test --automatic-sad --sad=/people/yin/projects/online_clustering/spotting_test/AMI.SpeakerSpotting.MixHeadset.test.mdtm /vol/work1/bredin/for_ruiqing/odyssey_llss/3s/ AMI.SpeakerSpotting.MixHeadset  result/3s/AMI.SpeakerSpotting.MixHeadset.segmentASAD.test.embedding.json

