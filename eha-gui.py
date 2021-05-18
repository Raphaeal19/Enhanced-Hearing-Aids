from tkinter import Tk, Label, PhotoImage, Button
import real_time_dtln_audio
import sounddevice as sd
import numpy as np

root = Tk()
noise_canc = real_time_dtln_audio.realtime_processing()
noise_off = sd.Stream(samplerate=noise_canc.fs_target, blocksize=noise_canc.block_shift,
                      dtype=np.float32, latency=noise_canc.latency,
                      channels=1, callback=noise_canc.noise_cancelation_callback)

noise_on = sd.Stream(samplerate=noise_canc.fs_target, blocksize=noise_canc.block_shift,
                     dtype=np.float32, latency=noise_canc.latency,
                     channels=1, callback=noise_canc.pass_through_callback)

root.title('Enhanced Hearing Aid')


root.geometry("1280x720")
is_on = False

my_label = Label(root,
                 text="Noise Cancellation is Off!",
                 fg="grey",
                 font=("Times", 32))

my_label.pack(pady=20)


def switch():
    global is_on

    if is_on:
        on_button.config(image=off)
        my_label.config(text="Noise Cancellation is Off!",
                        fg="grey")
        is_on = False
        noise_on.stop()
        noise_off.start()

    else:
        on_button.config(image=on)
        my_label.config(text="Noise Cancellation is On!", fg="green")
        is_on = True
        noise_off.stop()
        noise_on.start()


on = PhotoImage(file="./assests/on.png")
off = PhotoImage(file="./assests/off.png")

on_button = Button(root, image=off, bd=0,
                   command=switch)
on_button.pack(pady=50)

root.mainloop()
