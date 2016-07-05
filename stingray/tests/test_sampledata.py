
from ..sampledata import sample_data

class TestSampleData(object):
    """ Unit Tests for sampledata.py"""

    FIRST_LINE = [1.109110400703125000e+08, 4.120000061392784119e+03]
    LAST_LINE = [1.109138078203125000e+08, 2.619200039029121399e+03]
    FILE_LENGTH = 22143

    def test_file_exists(self):
        """ Test if file exists by checking the length
        of times and counts lists """
        lc = sample_data()
        assert len(lc.times) != 0
        assert len(lc.counts) != 0

    def test_file_first_line(self):
        """ Test if the first line matches with the
        actual data """
        lc = sample_data()
        assert lc.times[0] == self.FIRST_LINE[0]
        assert lc.counts[0] == self.FIRST_LINE[1]

    def test_file_last_line(self):
        """ Test if last line matches with the
        actual data """
        lc = sample_data()
        assert lc.times[-1] == self.LAST_LINE[0]
        assert lc.counts[-1] == self.LAST_LINE[1]

    def test_file_length(self):
        """ Test if file length is equal to actual
         length """
        lc = sample_data()
        assert len(lc.times) == self.FILE_LENGTH
        assert len(lc.counts) == self.FILE_LENGTH
