import numpy as np
import mmh3

class CountMinSketch:
    
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.sketch_array = np.zeros((hash_count,size),dtype=np.int)
        
    def add(self, string):
        for seed in xrange(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            self.sketch_array[seed][result]+=1
            
    def estimate(self, string):
        minimum = 1000000
        for seed in xrange(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            minimum = min(minimum,self.sketch_array[seed][result])
        return minimum
