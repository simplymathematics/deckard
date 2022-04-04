import unittest
from data import Data
datasets = ['mnist', 'cifar10']
# TODO other datasets
class testData(unittest.TestCase):
    
    def test_init(self):
        """
        Validates data object.
        """
        for dataset in datasets:
            data = Data(dataset)
            self.assertIsInstance(data, Data)
            self.assertIsInstance(data.params, dict)
            if isinstance(data.params, dict):
                self.assertIsInstance(data.params['dataset'], str)
                self.assertIsInstance(data.params['train_size'], (float, int))
                self.assertIsInstance(data.params['random_state'], int)
                self.assertIsInstance(data.params['shuffle'], bool)
                if data.params['target'] is not None:
                    self.assertIsInstance(data.params['target'], str)
                self.assertIsInstance(data.params['time_series'], bool)
            self.assertEqual(len(data.X_train), len(data.y_train))
            self.assertEqual(len(data.X_test), len(data.y_test))
    
    def test_hash(self):
        data1 = Data('mnist')
        data2 = Data('mnist')
        data3 = Data('cifar10')
        self.assertEqual(data1.__hash__(), data2.__hash__())
        self.assertNotEqual(data3.__hash__(), data2.__hash__())
    
    def test_eq(self):
        data1 = Data('mnist')
        data2 = Data('mnist')
        data3 = Data('cifar10')
        self.assertEqual(data1, data2)
        self.assertNotEqual(data1, data3)
    
    def test_get_params(self):
        data = Data('mnist')
        self.assertIsInstance(data.params['dataset'], str)
        self.assertIsInstance(data.params['train_size'], (float, int))
        self.assertIsInstance(data.params['random_state'], int)
        self.assertIsInstance(data.params['shuffle'], bool)
        if data.params['target'] is not None:
            self.assertIsInstance(data.params['target'], str)
        self.assertIsInstance(data.params['time_series'], bool)
    
    def test_set_params(self):
        data = Data('mnist')
        self.assertEqual(data.params['dataset'], 'mnist')
        data.set_params({'dataset': 'cifar10'})
        self.assertEqual(data.dataset, 'cifar10')
        data.set_params({'foo': 'bar'})
        self.assertEqual(data.params['foo'], 'bar')

    def test_choose_data(self):
        data = Data('mnist', random_state = 220)
        old = str(data.X_train)
        data._choose_data('cifar10')
        new = str(data.X_train)
        self.assertEqual(data.dataset, 'cifar10')
        self.assertNotEqual(old, new)
        data2 = Data('cifar10', random_state = 284)
        self.assertNotEqual(str(data.X_train), str(data2.X_train[1]))
    
    def test_parse_data(self):
        dataset = "https://raw.githubusercontent.com/simplymathematics/datasets/master/titanic.csv"
        data = Data(dataset, target = 'Survived', shuffle = False, stratify = False)
        self.assertIsInstance(data, Data)
        data2 = Data(dataset, target = 'Ticket', shuffle = False, stratify = False)
        self.assertIsInstance(data2, Data)
        self.assertNotEqual(data, data2)

    
