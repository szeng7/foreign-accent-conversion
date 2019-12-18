import PySimpleGUI as sg
import pyaudio
import wave

sg.change_look_and_feel('DarkAmber')	# Add a touch of color

# All the stuff inside your window.
layout = [  [sg.Text('Sound Boi')],
            [sg.Input(), sg.FileBrowse('Where it at tho?')],
            [sg.OK('Play WAV File'), sg.Cancel('Yeet out of here')]]

window = sg.Window('Get filename example', layout)

event, values = window.Read()

wavFilePath = values[0]

print('Path to wav file:', wavFilePath)

#define stream chunk
chunk = 1024

#open a wav format music
f = wave.open(wavFilePath,"rb")
#instantiate PyAudio
p = pyaudio.PyAudio()
#open stream
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                channels = f.getnchannels(),
                rate = f.getframerate(),
                output = True)
#read data
data = f.readframes(chunk)

#play stream
while data:
    stream.write(data)
    data = f.readframes(chunk)

#stop stream
stream.stop_stream()
stream.close()

#close PyAudio
p.terminate()

window.close()
