#!/usr/bin/env python

import MusicContent_basedApplication
import PySimpleGUI as sg

def choose_music():
    return MusicContent_basedApplication.df_meta['title'].sample(10).to_list()

sg.change_look_and_feel('GreenTan')

layout = [[sg.Text('Your name')],
    [sg.Input(justification='center',size=(25,1),key = 'userid')],
    [sg.Text('Choose 3 favorite songs')],
          [sg.InputCombo((choose_music()), size=(20, 1), key='song1')],
          [sg.InputCombo((choose_music()), size=(20, 1), key='song2')],
          [sg.InputCombo((choose_music()), size=(20, 1), key='song3')],
          [sg.Button('Show Recommend Songs'), sg.Exit()],
[sg.Text('Here are the songs recommended for you!', justification='center',size=(30,30), key='-OUTPUT-')]]

window = sg.Window('Everything bagel', layout,
    default_element_size=(40, 1), grab_anywhere=False)
while True:
    event, values = window.Read()
    if event is None or event == 'Exit':
        break

    if event == 'Show Recommend Songs':
        MusicsSelected = [values['song1']+values['song2']+values['song3']]
        window['-OUTPUT-'].Update(value=MusicContent_basedApplication.NewUser(MusicsSelected,values['userid'])['title'].to_list())

