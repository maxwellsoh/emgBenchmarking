import numpy as np
import struct
import datetime
from numpy import distutils
import mne

# from Poly5_Reader_MNE by Rudnik-Ilia

class Poly5Reader:
    def __init__(self, filename=None, readAll=True):
        self.filename = filename
        self.readAll = readAll
        self._readFile(filename)

    def _readFile(self, filename):
        try:
            self.file_obj = open(filename, "rb")
            file_obj = self.file_obj
            try:
                self._readHeader(file_obj)
                self.channels = self._readSignalDescription(file_obj)
                self._myfmt = 'f' * self.num_channels * self.num_samples_per_block
                self._buffer_size = self.num_channels * self.num_samples_per_block

                if self.readAll:
                    sample_buffer = np.zeros(self.num_channels * self.num_samples)

                    for i in range(self.num_data_blocks):
                        print('\rProgress: % 0.1f %%' % (100 * i / self.num_data_blocks), end="\r")
                        data_block = self._readSignalBlock(file_obj, self._buffer_size, self._myfmt)
                        i1 = i * self.num_samples_per_block * self.num_channels
                        i2 = (i + 1) * self.num_samples_per_block * self.num_channels
                        sample_buffer[i1:i2] = data_block

                    samples = np.transpose(
                        np.reshape(sample_buffer, [self.num_samples_per_block * (i + 1), self.num_channels]))
                    self.samples = samples
                    self.file_obj.close()
            except:
                print('Reading data failed.')
        except:
            print('Could not open file.')

    def readSamples(self, n_blocks=None):
        if n_blocks == None:
            n_blocks = self.num_data_blocks

        sample_buffer = np.zeros(self.num_channels * n_blocks * self.num_samples_per_block)

        for i in range(n_blocks):
            data_block = self._readSignalBlock(self.file_obj, self._buffer_size, self._myfmt)
            i1 = i * self.num_samples_per_block * self.num_channels
            i2 = (i + 1) * self.num_samples_per_block * self.num_channels
            sample_buffer[i1:i2] = data_block

        samples = np.transpose(np.reshape(sample_buffer, [self.num_samples_per_block * (i + 1), self.num_channels]))
        return samples

    def _readHeader(self, f):
        header_data = struct.unpack("=31sH81phhBHi4xHHHHHHHiHHH64x", f.read(217))
        magic_number = str(header_data[0])
        version_number = header_data[1]
        self.sample_rate = header_data[3]
        # self.storage_rate=header_data[4]
        self.num_channels = header_data[6] // 2
        self.num_samples = header_data[7]
        self.start_time = datetime.datetime(header_data[8], header_data[9], header_data[10], header_data[12], header_data[13], header_data[14])
        self.num_data_blocks = header_data[15]
        self.num_samples_per_block = header_data[16]
        if magic_number != "b'POLY SAMPLE FILEversion 2.03\\r\\n\\x1a'":
            print('This is not a Poly5 file.')
        elif version_number != 203:
            print('Version number of file is invalid.')
        else:
            print('\t Number of samples:  %s ' % self.num_samples)
            print('\t Number of channels:  %s ' % self.num_channels)
            print('\t Sample rate: %s Hz' % self.sample_rate)

    def _readSignalDescription(self, f):
        chan_list = []
        for ch in range(self.num_channels):
            channel_description = struct.unpack("=41p4x11pffffH62x", f.read(136))
            name = channel_description[0][5:].decode('ascii')
            unit_name = channel_description[1].decode('utf-8')
            '''
            ch = Channel(name, unit_name)
            chan_list.append(ch)
            '''
            chan_list.append(name)
            f.read(136)
        return chan_list

    def _readSignalBlock(self, f, buffer_size, myfmt):
        f.read(86)
        sampleData = f.read(buffer_size * 4)
        DataBlock = struct.unpack(myfmt, sampleData)
        SignalBlock = np.asarray(DataBlock)
        return SignalBlock

    def close(self):
        self.file_obj.close()


class Channel:
    def __init__(self, name, unit_name):
        self.__unit_name = unit_name
        self.__name = name
        self.n = name

if __name__ == "__main__":
    data=Poly5Reader(r'D:\ЗАГРУЗКИ\DASA\Session1\DASA_Resting_1_1_20181109_030747.Poly5')
    # print(type(data))
    # data.close()
    #
    # number_of_samples =  670428
    # number_of_channels =  31
    # sample_rate = 2048
    #
    # info = mne.create_info(number_of_channels, sfreq=sample_rate)
    # dataArray = np.array([number_of_channels,number_of_samples])
    # print(type(dataArray))
    # raw = mne.io.RawArray(data.samples,info)
    # print(raw)
    # # raw.plot(show_scrollbars=False, show_scalebars=False)
    # raw.plot()
    print(data._)