import PySimpleGUI as sg
import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "BigBoiVoice.wav"

def record():
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

sg.change_look_and_feel('DarkAmber')	# Add a touch of color

layout = [[sg.Button('Record my beautiful voice'), sg.Exit()] ]

window = sg.Window('ORIGINAL', layout)

while True:             # Event Loop
    event, values = window.Read()
    if event in (None, 'Exit'):
        break
    if event == 'Record my beautiful voice':
        record()
window.Close()



#
# # All the stuff inside your window.
# layout = [  [sg.Text('Sound Boi')],
#             [sg.Input(), sg.FileBrowse('Where it at tho?')],
#             [sg.OK('Play WAV File'), sg.Cancel('Yeet out of here')]]
#
# window = sg.Window('Get filename example', layout)
#
# event, values = window.Read()
#
# wavFilePath = values[0]
#
# print('Path to wav file:', wavFilePath)
# f = open("file_path.txt","w")
# f.write(wavFilePath)
# f.close()
#
# #define stream chunk
# chunk = 1024
#
# #open a wav format music
# f = wave.open(wavFilePath,"rb")
# #instantiate PyAudio
# p = pyaudio.PyAudio()
# #open stream
# stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
#                 channels = f.getnchannels(),
#                 rate = f.getframerate(),
#                 output = True)
# #read data
# data = f.readframes(chunk)
#
# #play stream
# while data:
#     stream.write(data)
#     data = f.readframes(chunk)
#
# #stop stream
# stream.stop_stream()
# stream.close()
#
# #close PyAudio
# p.terminate()
#
# window.close()
