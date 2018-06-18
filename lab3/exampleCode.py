"""
Code from Qin Lin at http://homepage.tudelft.nl/a57s4/TA/lab3.html


"""

# Frequent Algorithm - MISRA GRIES

def misra_gries(S, k):
    F_dict = {}
    for item in S:
        if item in F_dict.keys():
            F_dict[item] = F_dict[item]+1
        elif len(F_dict)<k-1:
            F_dict[item] = 1
        else:
            for jtem in F_dict.keys():
                F_dict[jtem] = F_dict[jtem]-1
                if F_dict[jtem] ==0:
                    if jtem in F_dict.keys(): del F_dict[jtem]
    return F_dict

data_stream = df['euro']
reservoirs = [10,100,1000]
query = [90, 72, 85, 38, 43, 75, 51, 81, 77, 76]#top 10 amount
for reservoirs_size in reservoirs:
    print 'reservoirs size ='+str(reservoirs_size)+':'
    result = misra_gries(data_stream,reservoirs_size)
    for jtem in query:
        if jtem in result.keys():
            print str(jtem)+': '+str(result[jtem])
        else:
            print 'None'
    print '**********'


# Bloom Filter
class BloomFilter:

    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, string):
        for seed in xrange(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            self.bit_array[result] = 1

    def lookup(self, string):
        for seed in xrange(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            if self.bit_array[result] == 0:
                return 0
        return 1
cutoff = int(0.5*len(data_stream))
hash_time = 10
data = data_stream.as_matrix()
train_data = data_stream[0:cutoff].as_matrix()
test_data = data_stream[cutoff:].as_matrix()
bf = BloomFilter(10000, hash_time)
for item in train_data:
    bf.add(str(item))
look_up_pre = np.zeros((len(test_data),), dtype=np.int)
look_up_real = np.zeros((len(test_data),), dtype=np.int)
# print test_data.as_matrix()
# print test_data.dtype
for i in xrange(len(test_data)):
    look_up_pre[i] = bf.lookup(str(test_data[i]))
for i in xrange(len(test_data)):
    look_up_real[i] = (test_data[i] in train_data)
look_up_real = look_up_real.astype(int)
TP, FP, FN, TN = 0, 0, 0, 0
for i in xrange(len(look_up_pre)):
    if look_up_real[i]==1 and look_up_pre[i]==1:
        TP += 1
    if look_up_real[i]==0 and look_up_pre[i]==1:
        FP += 1
    if look_up_real[i]==1 and look_up_pre[i]==0:
        FN += 1
    if look_up_real[i]==0 and look_up_pre[i]==0:
        TN += 1
print 'TP: '+ str(TP)
print 'FP: '+ str(FP)
print 'FN: '+ str(FN)
print 'TN: '+ str(TN)



# Count-Min Sketch
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
w = 1000#number of column
d = 10#number of row, hash count
cm = CountMinSketch(w, d)
query = [90, 72, 85, 38, 43, 75, 51, 81, 77, 76]#top 10 amount
for item in data:
    cm.add(str(item))
for item in query:
    frequency_est = cm.estimate(str(item))
    print str(item)+': '+str(frequency_est)
