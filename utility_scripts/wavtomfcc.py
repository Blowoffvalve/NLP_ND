from python_speech_features import mfcc
import scipy.io.wavfile as wav

def wavToMFCC(wav_filename, num_cepstrum):
    """ extract MFCC features from a wav file
    :param wav_filename: filename with a .wav format
    :param num_cepstrum: number of cepstrum to return
    :return: MFCC features for wav file
    """
    rate, data = wav.read(wav_filename)
    mfccFeatures = mfcc(data, rate, numcep=num_cepstrum)