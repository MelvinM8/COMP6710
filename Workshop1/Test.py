#Melvin Moreno
#COMP6710
#Workshop 1
#Date: 1/25/2023
#A simple test program that tests the calculator functions and handles exceptions.

import unittest
import Calc

class TestCalc(unittest.TestCase):
    def testMul(self):
        self.assertEqual(6, Calc.performMul(2, 3), "Multiplication failed")
        
    def testDiv(self):
        self.assertEqual(2, Calc.performDiv(6, 3), "Division failed")
    
if __name__ == '__main__':
    unittest.main()