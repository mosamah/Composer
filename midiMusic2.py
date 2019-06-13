# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:28:14 2018

@author: yousra
"""

"""
Frequency Note Table:
  Frequency   Note   MIDI#
    27.5000    A0     21
    29.1352    A#0    22
    30.8677    B0     23
    32.7032    C1     24
    34.6478    C#1    25   # C#1 = C1 * 1.059463094
    36.7081    D1     26   # 1.059463094 = 12th root of 2
    38.8909    D#1    27
    41.2034    E1     28
    43.6535    F1     29
    46.2493    F#1    30
    48.9994    G1     31
    51.9131    G#1    32
    55.0000    A1     33
    58.2705    A#1    34
    61.7354    B1     35
    65.4064    C2     36
    69.2957    C#2    37
    73.4162    D2     38
    77.7817    D#2    39
    82.4069    E2     40
    87.3071    F2     41
    92.4986    F#2    42
    97.9989    G2     43
   103.8262    G#2    44
   110.0000    A2     45
   116.5409    A#2    46
   123.4708    B2     47
   130.8128    C3     48
   138.5913    C#3    49
   146.8324    D3     50
   155.5635    D#3    51
   164.8138    E3     52
   174.6141    F3     53
   184.9972    F#3    54
   195.9977    G3     55
   207.6523    G#3    56
   220.0000    A3     57
   233.0819    A#3    58
   246.9417    B3     59
   261.6256    C4     60
   277.1826    C#4    61
   293.6648    D4     62
   311.1270    D#4    63
   329.6276    E4     64
   349.2282    F4     65
   369.9944    F#4    66
   391.9954    G4     67
   415.3047    G#4    68
   440.0000    A4     69
   466.1638    A#4    70
   493.8833    B4     71
   523.2511    C5     72
   554.3653    C#5    73
   587.3295    D5     74
   622.2540    D#5    75
   659.2551    E5     76
   698.4565    F5     77
   739.9888    F#5    78
   783.9909    G5     79
   830.6094    G#5    80
   880.0000    A5     81
   932.3275    A#5    82
   987.7666    B5     83
  1046.5023    C6     84
  1108.7305    C#6    85
  1174.6591    D6     86
  1244.5079    D#6    87
  1318.5102    E6     88
  1396.9129    F6     89
  1479.9777    F#6    90
  1567.9817    G6     91
  1661.2188    G#6    92
  1760.0000    A6     93
  1864.6550    A#6    94
  1975.5332    B6     95
  2093.0045    C7     96
  2217.4610    C#7    97
  2349.3181    D7     98
  2489.0159    D#7    99
  2637.0205    E7    100
  2793.8259    F7    101
  2959.9554    F#7   102
  3135.9635    G7    103
  3322.4376    G#7   104
  3520.0000    A7    105
  3729.3101    A#7   106
  3951.0664    B7    107
  4186.0090    C8    108
"""

from midiutil import MIDIFile



note_defs = { ("A4", 69),
             ("B4",71),
             ("C4",60),
             ("D4",62),
             ("E4",64),
             ("F4",65),
             ("G4",67)}


def playMidi(notes):
    track    = 0        #track: Each track is a list of messages
    channel  = 0
    time     = 0    # In beats
    volume   = 100  # 0-127, as per the MIDI standard
    tempo    = 120   # In BPM (beats per minute)
    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
                      
    MyMIDI.addTempo(track, time, tempo) 
    
    for note_name,note_type in notes:
        print(note_name,note_type)
        for note_midi in note_defs:
            if(note_name == note_midi[0]):
                midi_num = note_midi[1]
                break
            
        if(note_type == "whole"):
            duration = 4
        elif (note_type == "half"):
            duration = 2
        elif (note_type == "quarter"):
            duration = 1
        elif(note_type == "eigth"):
            duration = 0.5
        elif(note_type == "sixteenth"):
            duration = 0.25
            
        MyMIDI.addNote(track, channel, midi_num, time, duration, volume)
        time += duration
        
    with open("JingleBells.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)







                                   
notes = [("E4","quarter"),("B4","quarter"),("A4","quarter"),("G4","quarter"),("E4","half"),("E4","quarter"),
         ("B4","quarter"),("A4","quarter"),("G4","quarter"),("E4","half"),("E4","quarter"),("C4","quarter"),
         ("B4","quarter"),("A4","quarter")]

notes=[("B4","quarter"),("B4","quarter"),("B4","half"),("B4","quarter"),("B4","quarter"),("B4","half"),
       ("B4","quarter"),("D4","quarter"),("G4","quarter"),("A4","eight"),("B4","half")]
playMidi(notes)




