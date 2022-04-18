import pickle
import datetime
import time
import os
import traceback

class CheckpointOutputFileManager:
    def __init__(self, outputdir):
        self.outputdir = outputdir
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

    def getTimestamp(self):
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def getEmbeddingTag(self):
        return 'Embeddings'

    def getCnnTag(self):
        return 'CNN'

    def getLinearLayerTag(self):
        return 'Linear'

    def generateFilename(self, tag=''):
        timestamp = self.getTimestamp()
        if os.path.exists(os.path.join(self.outputdir, timestamp + '.pkl')):
            timestamp = timestamp + '_0'
        return '{}_{}.pkl'.format(tag, timestamp)

class Logger:
    def __init__(self, outputdir):
        self.file_manager = CheckpointOutputFileManager(outputdir)
        self.data = None

    def createDataPackage(self, data, tag=None):
        if tag is None:
            tag = 'data'
        self.data = { tag : data }

    def addDataToPackage(self, data, tag):
        if tag in self.data:
            tag = '{}_{}'.format(tag, self.file_manager.getTimestamp())
        self.data[tag] = data

    def writePackage(self, outputfilename, metadata=None):
        # Add metadata
        if metadata is None:
            metadata = { }
        if 'timestamp' not in metadata:
            metadata['timestamp'] = self.file_manager.getTimestamp()
        if 'user' not in metadata:
            metadata['user'] = os.getlogin()
        if 'stack_trace' not in metadata:
            metadata['stack_trace'] = traceback.format_stack()
        if 'metadata' in self.data:
            save_output = {'metadata' : metadata, 'data' : self.data}
        else:
            self.data['metadata'] = metadata
            save_output = self.data
        # Actually write the file
        pickle.dump(os.path.join(self.file_manager.outputdir, save_output), open(outputfilename, 'wb'))

    def loadPickle(self, filename):
        if not os.path.exists(filename):
            return pickle.load(open(os.path.join(self.file_manager.outputdir, filename), 'rb'))
        else:
            return pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    pass
