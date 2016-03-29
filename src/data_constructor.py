import pickle	
import pandas as pd
import json as js
import numpy as np

def attribute_generator(data):
	Attribute = {}
	for bussiness in data:
		for attri in bussiness['attributes'].keys():
			Attribute.setdefault(attri, [])
			if type(bussiness['attributes'][attri]) == dict:
				for attri2 in bussiness['attributes'][attri].keys():
					if attri2 not in Attribute[attri]:
						Attribute[attri].append(attri2)
	return Attribute

def categories_generator(data):
	Categories = []
	for bussiness in data:
		Categories += bussiness['categories']
	return list(set(Categories))

def construct_matrix(data, Attribute, Categories):
	'''
	The function will construct a DataFrame object which has the feature
	"review_count", "stars", "Categories", "Attribute"
	'''
	data1 = {}
	# review count & stars
	data1.setdefault("review_count",[])
	data1.setdefault("stars",[])
	data1.setdefault("business_id",[])
	# categories
	for cate in Categories:
		data1.setdefault("Categories_"+cate,[])
    
	for bussiness in data:
		data1["business_id"].append(bussiness["business_id"])
		data1["review_count"].append(bussiness["review_count"])
		data1["stars"].append(bussiness["stars"])
		for cate in Categories:
			if cate in bussiness["categories"]:
				data1["Categories_"+cate].append(1)
			else:
				data1["Categories_"+cate].append(0)
	
		for attri in Attribute.keys():
			if attri in bussiness["attributes"].keys():
				if len(Attribute[attri]) > 1:
					for attri2 in Attribute[attri]:
						data1.setdefault("Attribute_"+attri+"_"+attri2, [])
						if attri2 in bussiness["attributes"][attri].keys():
							data1["Attribute_"+attri+"_"+attri2].append(bussiness["attributes"][attri][attri2])
						else:
							data1["Attribute_"+attri+"_"+attri2].append(0)
				else:
					data1.setdefault("Attribute_"+attri, [])
					data1["Attribute_"+attri].append(bussiness["attributes"][attri])
			else:
				if len(Attribute[attri]) > 1:
					for attri2 in Attribute[attri]:
						data1.setdefault("Attribute_"+attri+"_"+attri2, [])
						data1["Attribute_"+attri+"_"+attri2].append(0)
				else:
					data1.setdefault("Attribute_"+attri, [])
					data1["Attribute_"+attri].append(0)
    
	return pd.DataFrame(data1)


	