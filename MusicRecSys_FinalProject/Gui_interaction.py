import Music_recsys_application
import numpy as np
import pandas as pd
import PySimpleGUI as sg

def choose_music():
    return Music_recsys_application.df_meta['title'].to_list()

sg.change_look_and_feel('GreenTan')
headings = ['contentBased', 'itemBased','userbased','ALSModel','SVDModel']
layout = [[sg.Text('Your name')],
    [sg.Input(justification='center',size=(25,1),key = 'userid')],
    [sg.Text('Choose 3 favorite songs')],
          [sg.InputCombo((choose_music()), size=(20, 1), key='song1')],
          [sg.InputCombo((choose_music()), size=(20, 1), key='song2')],
          [sg.InputCombo((choose_music()), size=(20, 1), key='song3')],
          [sg.Button('Show Recommend Songs'), sg.Exit()],
[sg.Text('Here are the songs recommended for you!', justification='center',size=(30,30), key='-OUTPUT-')],]

window = sg.Window('Everything bagel', layout,
    default_element_size=(40, 1), grab_anywhere=False)
while True:
    event, values = window.Read()
    if event is None or event == 'Exit':
        break

    if event == 'Show Recommend Songs':
        MusicsSelected = [values['song1'], values['song2'], values['song3']]
        CBF_contentBased, CF_itemBased, CF_userbased, CF_ALSModel, CF_SVDModel, df_user, df_user_final = Music_recsys_application.NewUser(MusicsSelected, values['userid'])
        CBF_contentBased_rec_songs = Music_recsys_application.get_title(CBF_contentBased.index)
        CF_itemBased_rec_songs = Music_recsys_application.get_title(CF_itemBased.index)
        CF_userbased_rec_songs = Music_recsys_application.get_title(CF_userbased.index)
        CF_ALSModel_rec_songs = Music_recsys_application.get_title(CF_ALSModel.index)
        CF_SVDModel_rec_songs = Music_recsys_application.get_title(CF_SVDModel.index)
        print(CBF_contentBased_rec_songs)
        df_rec = pd.DataFrame({'1': CBF_contentBased_rec_songs,
                      '2': CF_itemBased_rec_songs,
                      '3': CF_userbased_rec_songs,
                      '4': CF_ALSModel_rec_songs,
                      '5': CF_SVDModel_rec_songs})

        data = np.array(df_rec).tolist()
        layout2 = [[sg.Table(data, headings=headings, justification='left', key='-TABLE-')],]
        window = sg.Window("Title", layout2, finalize=True)
        while True:
            event, values = window.read()
            if event == sg.WINDOW_CLOSED:
                break
            print(event, values)
