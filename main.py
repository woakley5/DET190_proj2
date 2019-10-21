
from __future__ import division

import re
import sys
import os
import numpy as np

from google.cloud import speech
from google.cloud import language
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio #for recording audio!
import pygame  #for playing audio
from six.moves import queue
import wave
import wavio
from lights import Lights

from pydub import AudioSegment
from pydub.playback import play

from gtts import gTTS
import os
import time
from adafruit_crickit import crickit
from adafruit_seesaw.neopixel import NeoPixel


num_pixels = 75

#########################
sample_format = pyaudio.paFloat32  # 16 bits per sample
channels = 1
fs = 48000
seconds = 3
counter = 1
#########################

# The following line sets up a NeoPixel strip on Seesaw pin 20 for Feather
lights = Lights(num_pixels)

# Audio recording parameters, set for our USB mic.
RATE = 48000 #if you change mics - be sure to change this :)
chunk = CHUNK = int(RATE / 10)  # Record in chunks of 1024 samples
# chunk = CHUNK = 10  # Record in chunks of 1024 samples

# CHUNK = int(RATE / 10)  # 100ms
# CHUNK = 1024

credential_path = "/home/pi/Desktop/gcp_credentials.json" #replace with your file name!
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

client = speech.SpeechClient()
sentiment_client = language.LanguageServiceClient()


pygame.init()
pygame.mixer.init()

frame = []
final_string = ""

#MicrophoneStream() is brought in from Google Cloud Platform
class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.frames = []
        self.counter = 1


    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer
        )

        # self.frames = []

        # for i in range(0, int(RATE / CHUNK * 3)):
        #     data = self._audio_stream.read(CHUNK)
        #     self.frames.append(data)

        self.closed = False
        return self


    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.

        print("0")
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""

        # for i in range(0, int(fs / chunk * 3)):
        #     data = stream.read(chunk)
        frame.append(in_data)
        self._buff.put(in_data)
        return None, pyaudio.paContinue


    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

    def close(self):
        waveFile = wave.open("output" + str(self.counter) + ".wav", 'wb')
        self.counter += 1
        waveFile.setnchannels(channels)
        waveFile.setsampwidth(self._audio_interface.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frame))
        waveFile.close()

# mic = MicrophoneStream(RATE, CHUNK)


# #this loop is where the microphone stream gets sent
def listen_print_loop(responses, mic):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    final_string = ""

    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript
        print("transcript:" + transcript)

        final_string += transcript + " "

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
#            sys.stdout.write(transcript + overwrite_chars + '\r')
#            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)


            #if there's a voice activitated quit - quit!
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break
            if re.search(r'\b(save|send|night)\b', transcript, re.I):
                print("a")


                mic.close()
                decide_action(transcript, final_string)
            # else:
                # decide_action(transcript)
#            print(transcript)
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            num_chars_printed = 0


def decide_action(transcript, final_string):

    print("0")

    #set up the mixer
    freq = 44100     # audio CD quality
    bitsize = -16    # unsigned 16 bit
    channels = 2     # 1 is mono, 2 is stereo
    buffer = 2048    # number of samples (experiment to get right sound)
    pygame.mixer.init(freq, bitsize, channels, buffer)

    score = sentiment_analysis(final_string)

    print('string=' + transcript)
    print("1")
    print(score)

    song = AudioSegment.from_mp3("output1.wav")    
    louder_song = song + 20
    # play(louder_song)
    louder_song.export("output2.wav", format="wav")

    print("done")

    if score >= 0.1:
        print("positive")
        music_file1 = "music/positive/1.wav"
        music_file2 = "output2.wav"

        #Create sound object for each Audio
        myAudio1 = pygame.mixer.Sound(music_file1)
        myAudio2 = pygame.mixer.Sound(music_file2)
        #Add Audio to  first channel
        myAudio1.set_volume(0.8)
        myAudio2.set_volume(1.0)
        print("Playing audio : ", music_file1)

        pygame.mixer.Channel(0).play(myAudio2)
        pygame.mixer.Channel(1).play(myAudio1)


    elif score >= -0.1 and score <=0.1:
        print("neutral")
        music_file1 = "music/neutral/1.wav"
        music_file2 = "output2.wav"

        #Create sound object for each Audio
        myAudio1 = pygame.mixer.Sound(music_file1)
        myAudio2 = pygame.mixer.Sound(music_file2)
        #Add Audio to  first channel
        myAudio1.set_volume(0.8)
        myAudio2.set_volume(1.0)
        print("Playing audio: ", music_file1)

        pygame.mixer.Channel(0).play(myAudio2)
        pygame.mixer.Channel(1).play(myAudio1)


    elif score <= -0.1:
        print("negative")
        music_file1 = "music/negative/1.wav"
        music_file2 = "output2.wav"

        #Create sound object for each Audio
        myAudio1 = pygame.mixer.Sound(music_file1)
        myAudio2 = pygame.mixer.Sound(music_file2)
        #Add Audio to  first channel
        myAudio1.set_volume(0.8)
        myAudio2.set_volume(1.0)
        print("Playing audio: ", music_file1)

        pygame.mixer.Channel(0).play(myAudio2)
        pygame.mixer.Channel(1).play(myAudio1)


def intro():
    #set up the mixer
    freq = 44100     # audio CD quality
    bitsize = -16    # unsigned 16 bit
    channels = 2     # 1 is mono, 2 is stereo
    buffer = 2048    # number of samples (experiment to get right sound)
    pygame.mixer.init(freq, bitsize, channels, buffer)

    moon = "music/narration/moon_cropped.wav"
    tina = "music/narration/tina.wav"
    intro_voice = "music/narration/intro.wav"
    record_response = "music/narration/record_response.wav"
    time.sleep(1)

    myAudio1 = pygame.mixer.Sound(moon)
    myAudio2 = pygame.mixer.Sound(tina)
    myAudio3 = pygame.mixer.Sound(intro_voice)
    myAudio4 = pygame.mixer.Sound(record_response)

    pygame.mixer.Channel(2).play(myAudio3)

    time.sleep(8)

    print("Playing audio : ", myAudio1)

    myAudio1.set_volume(0.5)
    pygame.mixer.Channel(0).play(myAudio2)
    myAudio1.set_volume(0.5)
    pygame.mixer.Channel(1).play(myAudio1)
    time.sleep(17)

    print("Record response now : ", myAudio4)
    pygame.mixer.Channel(3).play(myAudio4)

    time.sleep(2)


def sentiment_analysis(text_to_analyze):
    document = language.types.Document(
        content=text_to_analyze,
        type=language.enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = sentiment_client.analyze_sentiment(document=document).document_sentiment
    return sentiment.score


def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    # this code comes from Google Cloud's Speech to Text API!
    # Check out the links in your handout. Comments are ours.
    language_code = 'en-US'  # a BCP-47 language tag
    # intro()

    #set up a client
    #make sure GCP is aware of the encoding, rate
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    #our example uses streamingrecognition - most likely what you will want to use.
    #check out the simpler cases of asychronous recognition too!
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    #this section is where the action happens:
    #a microphone stream is set up, requests are generated based on
    #how the audiofile is chunked, and they are sent to GCP using
    #the streaming_recognize() function for analysis. responses
    #contains the info you get back from the API.
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        #### Save audio recording ####

        ######
        # Now, put the transcription responses to use.
        listen_print_loop(responses, stream)
        # print('SAVING FILE')
        # # Save the recorded data as a WAV file
        # wavio.write("output4.wav", responses, fs, sampwidth=2)
        # print('Saved.')


if __name__ == '__main__':
    main()



# from __future__ import division

# import re
# import sys
# import os
# import numpy as np

# from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
# import pyaudio #for recording audio!
# import pygame  #for playing audio
# from six.moves import queue
# import wave


# from gtts import gTTS
# import os
# import time
# from adafruit_crickit import crickit
# from adafruit_seesaw.neopixel import NeoPixel

# num_pixels = 75  # Number of pixels driven from Crickit NeoPixel terminal

# #########################
# chunk = 1024  # Record in chunks of 1024 samples
# sample_format = pyaudio.paInt16  # 16 bits per sample
# channels = 2
# fs = 48000
# seconds = 3
# filename = "output.wav"
# #########################

# # The following line sets up a NeoPixel strip on Seesaw pin 20 for Feather
# pixels = NeoPixel(crickit.seesaw, 20, num_pixels)

# # Audio recording parameters, set for our USB mic.
# RATE = 48000 #if you change mics - be sure to change this :)
# CHUNK = int(RATE / 10)  # 100ms

# credential_path = "/home/pi/Desktop/gcp_credentials.json" #replace with your file name!
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=credential_path

# client = speech.SpeechClient()

# pygame.init()
# pygame.mixer.init()

# #MicrophoneStream() is brought in from Google Cloud Platform
# class MicrophoneStream(object):
#     """Opens a recording stream as a generator yielding the audio chunks."""
#     def __init__(self, rate, chunk):
#         self._rate = rate
#         self._chunk = chunk

#         # Create a thread-safe buffer of audio data
#         self._buff = queue.Queue()
#         self.closed = True

#     def __enter__(self):
#         self._audio_interface = pyaudio.PyAudio()
#         self._audio_stream = self._audio_interface.open(
#             format=pyaudio.paInt16,
#             # The API currently only supports 1-channel (mono) audio
#             # https://goo.gl/z757pE
#             channels=1, rate=self._rate,
#             input=True, frames_per_buffer=self._chunk,
#             # Run the audio stream asynchronously to fill the buffer object.
#             # This is necessary so that the input device's buffer doesn't
#             # overflow while the calling thread makes network requests, etc.
#             stream_callback=self._fill_buffer,
#         )

#         self.closed = False

#         return self

#     def __exit__(self, type, value, traceback):
#         self._audio_stream.stop_stream()
#         self._audio_stream.close()
#         self.closed = True
#         # Signal the generator to terminate so that the client's
#         # streaming_recognize method will not block the process termination.
#         self._buff.put(None)
#         self._audio_interface.terminate()

#     def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
#         """Continuously collect data from the audio stream, into the buffer."""
#         self._buff.put(in_data)
#         return None, pyaudio.paContinue

#     def generator(self):
#         while not self.closed:
#             # Use a blocking get() to ensure there's at least one chunk of
#             # data, and stop iteration if the chunk is None, indicating the
#             # end of the audio stream.
#             chunk = self._buff.get()
#             if chunk is None:
#                 return
#             data = [chunk]

#             # Now consume whatever other data's still buffered.
#             while True:
#                 try:
#                     chunk = self._buff.get(block=False)
#                     if chunk is None:
#                         return
#                     data.append(chunk)
#                 except queue.Empty:
#                     break

#             yield b''.join(data)

# # #this loop is where the microphone stream gets sent
# def listen_print_loop(responses):
#     """Iterates through server responses and prints them.

#     The responses passed is a generator that will block until a response
#     is provided by the server.

#     Each response may contain multiple results, and each result may contain
#     multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
#     print only the transcription for the top alternative of the top result.

#     In this case, responses are provided for interim results as well. If the
#     response is an interim one, print a line feed at the end of it, to allow
#     the next result to overwrite it, until the response is a final one. For the
#     final one, print a newline to preserve the finalized transcription.
#     """
#     num_chars_printed = 0
#     for response in responses:
#         if not response.results:
#             continue

#         # The `results` list is consecutive. For streaming, we only care about
#         # the first result being considered, since once it's `is_final`, it
#         # moves on to considering the next utterance.
#         result = response.results[0]
#         if not result.alternatives:
#             continue

#         # Display the transcription of the top alternative.
#         transcript = result.alternatives[0].transcript
#         print("transcript:" + transcript)

#         # Display interim results, but with a carriage return at the end of the
#         # line, so subsequent lines will overwrite them.
#         #
#         # If the previous result was longer than this one, we need to print
#         # some extra spaces to overwrite the previous result
#         overwrite_chars = ' ' * (num_chars_printed - len(transcript))

#         if not result.is_final:
# #            sys.stdout.write(transcript + overwrite_chars + '\r')
# #            sys.stdout.flush()

#             num_chars_printed = len(transcript)

#         else:
#             print(transcript + overwrite_chars)


#             #if there's a voice activitated quit - quit!
#             if re.search(r'\b(exit|quit)\b', transcript, re.I):
#                 print('Exiting..')
#                 break
#             else:
#                 decide_action(transcript)
# #            print(transcript)
#             # Exit recognition if any of the transcribed phrases could be
#             # one of our keywords.
#             num_chars_printed = 0

# def decide_action(transcript):

#     #here we're using some simple code on the final transcript from
#     #GCP to figure out what to do, how to respond.

#     if re.search('test',transcript, re.I):
#         fillLEDs((0,10,10))
#     elif re.search('off',transcript, re.I):
#         fillLEDs((0,0,0))



# def fillLEDs(color):

#     pixels.fill(color)


# def main():
#     # See http://g.co/cloud/speech/docs/languages
#     # for a list of supported languages.
#     # this code comes from Google Cloud's Speech to Text API!
#     # Check out the links in your handout. Comments are ours.
#     language_code = 'en-US'  # a BCP-47 language tag

#     #set up a client
#     #make sure GCP is aware of the encoding, rate
#     config = types.RecognitionConfig(
#         encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=RATE,
#         language_code=language_code)
#     #our example uses streamingrecognition - most likely what you will want to use.
#     #check out the simpler cases of asychronous recognition too!
#     streaming_config = types.StreamingRecognitionConfig(
#         config=config,
#         interim_results=True)

#     #this section is where the action happens:
#     #a microphone stream is set up, requests are generated based on
#     #how the audiofile is chunked, and they are sent to GCP using
#     #the streaming_recognize() function for analysis. responses
#     #contains the info you get back from the API.
#     with MicrophoneStream(RATE, CHUNK) as stream:
#         audio_generator = stream.generator()
#         requests = (types.StreamingRecognizeRequest(audio_content=content)
#                     for content in audio_generator)

#         responses = client.streaming_recognize(streaming_config, requests)

#         #### Save audio recording ####

#         ######
#         # Now, put the transcription responses to use.
#         listen_print_loop(responses)
#         print('SAVING FILE')
#         # Save the recorded data as a WAV file
#         wavio.write("output.wav", responses, fs, sampwidth=2)
#         print('Saved.')




# def recording():
#     p = pyaudio.PyAudio()  # Create an interface to PortAudio
#     print('Recording')
#     stream = p.open(format=sample_format,
#                     channels=channels,
#                     rate=fs,
#                     frames_per_buffer=chunk,
#                     input=True)

#     frames = []  # Initialize array to store frames

#     # Store data in chunks for 3 seconds
#     for i in range(0, int(fs / chunk * seconds)):
#         data = stream.read(chunk)
#         frames.append(data)


#     print('Finished recording')

#     # Save the recorded data as a WAV file
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(p.get_sample_size(sample_format))
#     wf.setframerate(fs)
#     wf.writeframes(b''.join(frames))
#     wf.close()


# if __name__ == '__main__':
#     main()
