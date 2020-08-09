"""
These functions are used for translating between midi and the binary piano roll format,
as well as synthesizing new music.
"""

import numpy as np
import random
import torch
import mido
import os


def get_min_max_note(directory: str):
    """
    :param directory: name of directory containing a bunch of midi files
    :return: minimum and maximum note found in all files
    """

    min_note = 255
    max_note = 0

    for filename in os.listdir(directory):
        if filename.endswith('.mid'):
            midfile = mido.MidiFile(directory + '/' + filename)
            for msg in midfile:
                if msg.type == 'note_on':
                    note = msg.note
                    if note > max_note:
                        max_note = note
                    if note < min_note:
                        min_note = note

    return min_note, max_note


def get_msg_list(midi):
    """
    :param midi: a parsed midi track
    :return: list of all on/off signals (filters meta signals, etc)
    """
    result = []
    for msg in midi:
        if msg.type == 'note_on' or msg.type == 'note_off':
            result.append(msg)
    return result


def get_increment(msg_list):
    """
    :param msg_list: list of on/off midi signals
    :return: minimum time of all signals
    """
    result = float('inf')
    for msg in msg_list:
        if msg.time != 0.0 and msg.time < result:
            result = msg.time
    return result


def to_piano_roll(in_filename: str, min_note: int, max_note: int):
    """
    :param in_filename: name of midi file to be converted to piano roll
    :param min_note: minimum note that might be contained in the file
    :param max_note: maximum note that might be contained in the file
    :return: binary numpy array whose first index is time and whose second index is note id - element will be 1 if the note is on at that time and 0 otherwise.
    """

    midi = mido.MidiFile(in_filename)
    msg_list = get_msg_list(midi)
    increment = get_increment(msg_list)

    array = np.zeros((int(midi.length/increment) + 1, max_note - min_note + 1), dtype='uint8')

    index = 0
    on_notes = []

    for msg in msg_list:

        time = int(msg.time/increment)

        if msg.type == 'note_on':
            on_notes.append((msg.note - min_note, msg.velocity))
        elif msg.type == 'note_off':
            on_notes = [(n, v) for (n, v) in on_notes if n != msg.note - min_note]

        for t in range(index, index + time + 1):
            for note, velocity in on_notes:
                array[t, note] = 1

        index += time

    return array


def to_midi(min_note, piano_roll_song, filename):

    # create a new midi file with a single track
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    #track.append(mido.Message('program_change', program=12, time=0))

    track.append(mido.MetaMessage('key_signature', key='C', time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    #track.append(mido.MetaMessage('set_tempo', tempo=705882))

    # keep track of which notes are being played at the given time
    on_notes = []

    # keep track of how many notes have passed since a change was made
    since_change = 1

    # record all messages since the last change
    # when a change occurs, append them with the last one indicating the amount of time passed
    old_messages = []
    new_messages = []

    # loop through time
    for i in range(piano_roll_song.shape[0]):

        # did anything change at this time step
        nothing_happened = True

        # if this is the last time step, that counts as something happening
        if i == piano_roll_song.shape[0] - 1:
            nothing_happened = False

        # loop through notes
        for j in range(piano_roll_song.shape[1]):

            # if this note is being played at the current time
            if piano_roll_song[i, j] == 1:

                # if it was not being played before
                if not j in on_notes:

                    # this new message indicates the note was turned on
                    new_msg = mido.Message('note_on', note=min_note+j, velocity=90, time=0)

                    # it is a new message so we append it to the appropriate list
                    new_messages.append(new_msg)

                    # something happened, this note was turned on
                    nothing_happened = False

                    # record that this note is being played
                    on_notes.append(j)

                # otherwise do nothing
                else:
                    continue

            # if this note is not being played at the current time
            else:

                # if it was being played
                if j in on_notes:

                    # this new message indicates the note was turned off
                    new_msg = mido.Message('note_off', note=min_note+j, velocity=0, time=0)

                    # it is a new message so we append it to the appropriate list
                    new_messages.append(new_msg)

                    # something happened, this note was turned off
                    nothing_happened = False

                    # remove the note from on_notes
                    on_notes = [note for note in on_notes if not note == j]

                # otherwise do nothing
                else:
                    continue

        # if nothing happened simply increment the time since something happened
        if nothing_happened:
            since_change += 1

        # if something happened, append all the messages that have been accumulating
        # and replace the old messages with the new ones
        else:

            # if this is the first time step there will be no old messages
            if len(old_messages) > 0:

                # make the last old message lag the appropriate time
                old_messages[-1].time = 120*since_change

                # append the old messages
                for msg in old_messages:
                    track.append(msg)

                # erase old messages and replace them with new, reset counter
                old_messages = new_messages
                new_messages = []
                since_change = 1

            else:

                # we need to wait to append them until we know how much time has passed
                old_messages = new_messages
                new_messages = []

    # save the file
    mid.save(filename)


def make_music(model,
               piano_roll,
               true_steps: int,
               input_steps: int,
               free_steps: int,
               history: int,
               variance: float):
    """
    :param model: pytorch model which will be used to synthesize new music
    :param piano roll: 2D binary array indicating which notes are played at a given time
    :param true_steps: how many frames of the new song are to be copied directly from the original
    :param input_steps: how many frames of the new song are to be the output of the model being fed piano_roll
    :param free_steps: how many frames of the new song are to be the model predicting the next frame based off of the song constructed so far
    :param history: how many time steps the model will look back in the past while constructing the new song
    :param variance: variance of the noise to be added to the hidden state (in order to avoid stable periodic orbits)
    """

    # first few steps of the song will be the original music
    T = true_steps + input_steps + free_steps
    song = np.zeros((T, 88), dtype='uint8')
    song[0 : true_steps] = piano_roll[0 : true_steps]

    # format the input to the model
    tsis = true_steps + input_steps
    input_tensor = torch.tensor(piano_roll[0 : tsis], dtype=torch.float)
    input_tensor = input_tensor.unsqueeze(0)

    # format the output of the model
    # the next few steps will be the output of the model given the true song as input
    output_tensor, hiddens = model(input_tensor)
    binary = (torch.sigmoid(output_tensor) > 0.5).type(torch.uint8)
    reformatted = binary.reshape(tsis, 88).detach().numpy()
    song[true_steps : true_steps + input_steps] = reformatted[true_steps : true_steps + input_steps]

    # some tricks to keep the system from falling into a periodic orbit
    last_binary = None
    periodicity_detected = False

    # the last steps of the model will be the model making predictions off of its own output
    for t in range(tsis, tsis + free_steps):

        # get the last frame of the new song, double the intensity since it is both input and last step
        last_output = 2*torch.tensor(song[t - 1 - history : t - 1], dtype=torch.float).unsqueeze(0)
        shp = last_output.shape
        last_output[:, :, 25 : 75] += variance*torch.randn((1, history, 50))

        # add some extra noise to the input to some channels to knock the system out of a limit cycle
        """
        if periodicity_detected:

            # randomly select 10 channels to add noise too
            for i in range(10):

                ix = random.randint(25, 75)
                last_output[:, :, ix] += 0.5*torch.randn((1, history))

            #last_output += 0.5*torch.randn((1, history, 88))

            periodicity_detected = False
        """

        # use the model to predict the next frame
        new_output, hiddens = model(last_output)
        binary = (torch.sigmoid(new_output[0, -1]) > 0.5).type(torch.uint8)
        reformatted = binary.reshape(88).detach().type(torch.uint8).numpy()

        # filter out note blasts
        if np.sum(reformatted) > 8:

            num = np.sum(reformatted, dtype='int') - 8
            ixs = [i for i in range(88) if reformatted[i] > 0]

            reformatted = np.zeros(reformatted.shape)

            for i in range(num//2, len(ixs) - num//2):
                reformatted[ixs[i]] = 1

        song[t] = reformatted

        now = song[t]
        two = song[t - 2]
        three = song[t - 3]
        four = song[t - 4]
        if np.sum(now*two) > 3 or np.sum(now*three) > 3 or np.sum(now*four) > 3:
            periodicity_detected = True

    return song