# Low-latency speaker spotting
Low-latency speaker spotting with online diarization and detection## Installation
```bash
$ conda create --name pyannote python=3.6 anaconda
$ source activate pyannote
$ conda install -c yaafe yaafe=0.65
$ pip install pyannote.audio
$ pip install pyannote.db.odessa.ami
$ git clone
```

## Citation
If you use this tool, please cite the following paper:
```
@inproceedings{patino2018low,
  title={{Low-latency Speaker Spotting with Online Diarization and Detection}},
  author={Patino, Jose and Yin, Ruiqing and Delgado, H\'ector and Bredin, Herv\'e and Komaty, Alain and Wisniewski, Guillaume and Barras, Claude and Evans, Nicholas and Marcel, S\'ebastien },
  booktitle={The Speaker and Language Recognition Workshop (Odyssey 2018)},
  year={2018}
}
```

## Usage
### Embedding training and extraction
Please follow the [`documentation `](https://github.com/pyannote/pyannote-audio/tree/master/tutorials/speaker-embedding) in [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) to extract embeddings. A pretrained model (trained and validated in AMI database) is available in 

### Low-latency speaker spotting with segmental diarization
```bash
$ export $EMBEDDING_DIR=/path/of/extracted/embeddings
$ conda create --name pyannote python=3.6 anaconda

```
### Other options 
```bash
$ export $EMBEDDING_DIR=/path/of/extracted/embeddings
$ conda create --name pyannote python=3.6 anaconda

```

