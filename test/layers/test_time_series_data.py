import unittest, tempfile, os, shutil
from deckard.base import Data
import numpy as np
dataset = '../../examples/time_series/US-CAL-CISO 2020.csv'
# TODO other datasets
class testTimeSeriesData(unittest.TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp()
       
    def test_init(self):
        """
        Validates data object.
        """
        data = Data(dataset, target = 'carbon_intensity_avg')
    
    def test_data_hash(self):
        data1 = Data(dataset, time_series = True)
        data2 = Data(dataset, time_series = True)
        data3 = Data('cifar10')
        self.assertEqual(data1.__hash__(), data2.__hash__())
        self.assertNotEqual(data3.__hash__(), data2.__hash__())
    
    def test_data_eq(self):
        data1 = Data(dataset, time_series = True)
        data2 = Data(dataset, time_series = True)
        data3 = Data('cifar10')
        self.assertEqual(data1, data2)
        self.assertNotEqual(data1, data3)
    
    def test_data_get_params(self):
        data = Data(dataset, time_series = True)
        self.assertIsInstance(data.params['dataset'], str)
        self.assertIsInstance(data.params['train_size'], (float, int))
        self.assertIsInstance(data.params['random_state'], int)
        self.assertIsInstance(data.params['shuffle'], bool)
        if data.params['target'] is not None:
            self.assertIsInstance(data.params['target'], str)
        self.assertIsInstance(data.params['time_series'], bool)
    
    def test_model_init(self):
        pass
     
    
    def tearDown(self):
        shutil.rmtree(self.path)
        
if __name__ == '__main__':
    unittest.main()