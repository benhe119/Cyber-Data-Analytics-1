import re

pcap_mask = re.compile('^(\d{4}[-]+\d{2}[-]+\d{2})\s*(\d{2}[:]+\d{2}[:]+[0-9.]{6})\s*([0-9.]{5})\s*(\w*)\s*([0-9]+(?:\.[0-9]+){3})[:]([0-9]+)\s*[->]*\s*([0-9]+(?:\.[0-9]+){3})[:]([0-9]+)\s*(\w*)\s*(\d*)\s*(\d*)\s*(\d*)\s*(\d*)\s*(\w*)')
source_file = 'data/' #Change to original file

extracted_data = [re.findall(pcap_mask, line) for line in open(source_file)]
extracted_data.pop(0) #remove header

data_matrix = [i for sub_list in extracted_data for i in sub_list]

print(data_matrix)
print(data_matrix[0]) #Get first line
print(data_matrix[0][3]) #Get fourth item of first line
