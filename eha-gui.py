from ttkbootstrap import Style
from tkinter import Scale, Tk, Label, PhotoImage, Button, DoubleVar, font
from tkinter.constants import CENTER, HORIZONTAL
import real_time_dtln_audio
import sounddevice as sd
import numpy as np

style = Style(theme='superhero')
style.configure('custom.TFrame', background='gray')

root = style.master
var = DoubleVar()
noise_canc = real_time_dtln_audio.realtime_processing()
noise_off = sd.Stream(samplerate=noise_canc.fs_target, blocksize=noise_canc.block_shift,
                      dtype=np.float32, latency=noise_canc.latency,
                      channels=1, callback=noise_canc.noise_cancelation_callback)

noise_on = sd.Stream(samplerate=noise_canc.fs_target, blocksize=noise_canc.block_shift,
                     dtype=np.float32, latency=noise_canc.latency,
                     channels=1, callback=noise_canc.pass_through_callback)

root.title('Enhanced Hearing Aid')


root.geometry("800x400")
root.minsize(800, 400)
is_on = False

our_font = font.Font(family="Comic Sans MS",
                     size=20,
                     weight="bold")

my_label = Label(root,
                 text="Noise Cancellation is Off!",
                 fg="grey",
                 font=our_font)

my_label.pack(pady=40)


def switch():
    global is_on

    if is_on:
        on_button.config(image=off)
        my_label.config(text="Noise Cancellation is Off!",
                        fg="grey")
        is_on = False
        noise_off.stop()
        noise_on.start()

    else:
        on_button.config(image=on)
        my_label.config(text="Noise Cancellation is On!", fg="green")
        is_on = True
        noise_on.stop()
        noise_off.start()


on = PhotoImage(file="./assests/on.png")
off = PhotoImage(file="./assests/off.png")

on_button = Button(root, image=off, bd=0,
                   command=switch)
on_button.pack(pady=30)

scale = Scale(root, variable=var, orient=HORIZONTAL,
              from_=0.1, resolution=0.1, to=1, showvalue=1)
scale.pack(anchor=CENTER)
vol_label = Label(root, text="Latency")
vol_label.pack()

root.mainloop()
