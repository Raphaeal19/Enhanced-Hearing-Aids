import time
import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


class realtime_processing:

    def __init__(self, latency=0.1):
        self.latency = latency

        self.block_len_ms = 32
        self.block_shift_ms = 8
        self.fs_target = 16000
        # create the interpreters
        self.interpreter_1 = tflite.Interpreter(
            model_path='./pretrained_model/model_1.tflite')
        self.interpreter_1.allocate_tensors()
        self.interpreter_2 = tflite.Interpreter(
            model_path='./pretrained_model/model_2.tflite')
        self.interpreter_2.allocate_tensors()
        # Get input and output tensors.
        self.input_details_1 = self.interpreter_1.get_input_details()
        self.output_details_1 = self.interpreter_1.get_output_details()
        self.input_details_2 = self.interpreter_2.get_input_details()
        self.output_details_2 = self.interpreter_2.get_output_details()
        # create states for the lstms
        self.states_1 = np.zeros(
            self.input_details_1[1]['shape']).astype('float32')
        self.states_2 = np.zeros(
            self.input_details_2[1]['shape']).astype('float32')
        # calculate shift and length
        self.block_shift = int(
            np.round(self.fs_target * (self.block_shift_ms / 1000)))
        self.block_len = int(
            np.round(self.fs_target * (self.block_len_ms / 1000)))
        # create buffer
        self.in_buffer = np.zeros((self.block_len)).astype('float32')
        self.out_buffer = np.zeros((self.block_len)).astype('float32')

    def noise_cancelation_callback(self, indata, outdata, frames, time, status):
        # buffer and states to global
        # write to buffer
        self.in_buffer[:-self.block_shift] = self.in_buffer[self.block_shift:]
        self.in_buffer[-self.block_shift:] = np.squeeze(indata)
        # calculate fft of input block
        in_block_fft = np.fft.rfft(self.in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
        # set tensors to the first model
        self.interpreter_1.set_tensor(
            self.input_details_1[1]['index'], self.states_1)
        self.interpreter_1.set_tensor(self.input_details_1[0]['index'], in_mag)
        # run calculation
        self.interpreter_1.invoke()
        # get the output of the first block
        out_mask = self.interpreter_1.get_tensor(
            self.output_details_1[0]['index'])
        self.states_1 = self.interpreter_1.get_tensor(
            self.output_details_1[1]['index'])
        # calculate the ifft
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        # reshape the time domain block
        estimated_block = np.reshape(
            estimated_block, (1, 1, -1)).astype('float32')
        # set tensors to the second block
        self.interpreter_2.set_tensor(
            self.input_details_2[1]['index'], self.states_2)
        self.interpreter_2.set_tensor(
            self.input_details_2[0]['index'], estimated_block)
        # run calculation
        self.interpreter_2.invoke()
        # get output tensors
        out_block = self.interpreter_2.get_tensor(
            self.output_details_2[0]['index'])
        self.states_2 = self.interpreter_2.get_tensor(
            self.output_details_2[1]['index'])
        # write to buffer
        self.out_buffer[:-
                        self.block_shift] = self.out_buffer[self.block_shift:]
        self.out_buffer[-self.block_shift:] = np.zeros((self.block_shift))
        self.out_buffer += np.squeeze(out_block)
        # output to soundcard
        outdata[:] = np.expand_dims(
            self.out_buffer[:self.block_shift], axis=-1)

    def pass_through_callback(self, indata, outdata, frames, time, status):
        outdata[:] = indata
