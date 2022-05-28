#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: record nfc with rtl-sdr
# GNU Radio version: v3.11.0.0git-104-g8ccb8e65

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import osmosdr
import time



from gnuradio import qtgui

class record_nfc(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "record nfc with rtl-sdr", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("record nfc with rtl-sdr")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "record_nfc")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.subport = subport = 847.5e3
        self.samp_rate = samp_rate = 1.8e6
        self.nfc_freq = nfc_freq = +13.56e6
        self.upconverter = upconverter = 125e6
        self.transition_width = transition_width = 10e3
        self.rf_gain = rf_gain = 49.6
        self.low_freq = low_freq = 1
        self.high_freq = high_freq = samp_rate/2
        self.freq = freq = nfc_freq
        self.cutoff_freq = cutoff_freq = subport/2
        self.const = const = 0

        ##################################################
        # Blocks
        ##################################################
        self._rf_gain_range = Range(0, 50, 1, 49.6, 200)
        self._rf_gain_win = RangeWidget(self._rf_gain_range, self.set_rf_gain, "'rf_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._rf_gain_win)
        self._freq_range = Range((nfc_freq-1.8e6), (nfc_freq+1.8e6*2), 1e2, nfc_freq, 200)
        self._freq_win = RangeWidget(self._freq_range, self.set_freq, "'freq'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._freq_win)
        self._transition_width_range = Range(1, 500e3, 1, 10e3, 200)
        self._transition_width_win = RangeWidget(self._transition_width_range, self.set_transition_width, "'transition_width'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._transition_width_win)
        self.rtlsdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + "rtl=0,direct_samp=2"
        )
        self.rtlsdr_source_0.set_sample_rate(samp_rate)
        self.rtlsdr_source_0.set_center_freq(freq, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_dc_offset_mode(0, 0)
        self.rtlsdr_source_0.set_iq_balance_mode(0, 0)
        self.rtlsdr_source_0.set_gain_mode(False, 0)
        self.rtlsdr_source_0.set_gain(rf_gain, 0)
        self.rtlsdr_source_0.set_if_gain(0, 0)
        self.rtlsdr_source_0.set_bb_gain(0, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
        self.rtlsdr_source_0.set_block_alias("rtl_sdr")
        self._low_freq_range = Range(1, samp_rate/2, 1e3, 1, 200)
        self._low_freq_win = RangeWidget(self._low_freq_range, self.set_low_freq, "'low_freq'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._low_freq_win)
        self._high_freq_range = Range(1, samp_rate/2, 1e3, samp_rate/2, 200)
        self._high_freq_win = RangeWidget(self._high_freq_range, self.set_high_freq, "'high_freq'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._high_freq_win)
        self._cutoff_freq_range = Range(1, samp_rate/2, 10e2, subport/2, 200)
        self._cutoff_freq_win = RangeWidget(self._cutoff_freq_range, self.set_cutoff_freq, "'cutoff_freq'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._cutoff_freq_win)
        self._const_range = Range(-1, 0, 0.1, 0, 200)
        self._const_win = RangeWidget(self._const_range, self.set_const, "'const'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._const_win)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/luca/github/university/nfc-unipd/data/gqrx_fixed/ground-sending.raw', False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.rtlsdr_source_0, 0), (self.blocks_file_sink_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "record_nfc")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_subport(self):
        return self.subport

    def set_subport(self, subport):
        self.subport = subport
        self.set_cutoff_freq(self.subport/2)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_high_freq(self.samp_rate/2)
        self.rtlsdr_source_0.set_sample_rate(self.samp_rate)

    def get_nfc_freq(self):
        return self.nfc_freq

    def set_nfc_freq(self, nfc_freq):
        self.nfc_freq = nfc_freq
        self.set_freq(self.nfc_freq)

    def get_upconverter(self):
        return self.upconverter

    def set_upconverter(self, upconverter):
        self.upconverter = upconverter

    def get_transition_width(self):
        return self.transition_width

    def set_transition_width(self, transition_width):
        self.transition_width = transition_width

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self.rtlsdr_source_0.set_gain(self.rf_gain, 0)

    def get_low_freq(self):
        return self.low_freq

    def set_low_freq(self, low_freq):
        self.low_freq = low_freq

    def get_high_freq(self):
        return self.high_freq

    def set_high_freq(self, high_freq):
        self.high_freq = high_freq

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.rtlsdr_source_0.set_center_freq(self.freq, 0)

    def get_cutoff_freq(self):
        return self.cutoff_freq

    def set_cutoff_freq(self, cutoff_freq):
        self.cutoff_freq = cutoff_freq

    def get_const(self):
        return self.const

    def set_const(self, const):
        self.const = const




def main(top_block_cls=record_nfc, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
