import pandas
import pickle
# sys.path.append("../tools/")

with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

data_dict.pop('TOTAL',0)
pd_data= pandas.DataFrame.from_dict(data_dict)
incompletes=dict()
for col in pd_data:
	if col!='poi':
		incompletes[col]=sum(pd_data[col]=='NaN')

pd_incompletes= pandas.DataFrame.from_dict(incompletes, orient='index')

print pd_incompletes[pd_incompletes[0]>=18]

print data_dict['WHALEY DAVID A']
#LOCKHART, EUGENE E