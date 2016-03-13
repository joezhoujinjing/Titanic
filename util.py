def clean_cabin(x):
	try:
		return x[0]
	except TypeError:
		return 'None'

def shuffle(df, n=1, axis=0):
	df = df.copy()
	axis = int(not axis)
	for _ in range(n):
		for view in np.rollaxis(df.values, axis):
			np.random.shuffle(view)
	return df

def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
    if autoscale:
        x_scale = model.feature_importances_.max()+headroom
    else:
        x_scale = 1
    feature_dict = dict(zip(feature_names,model.feature_importances_))
    if summarized_columns:
        for col_name in summarized_columns:
            sum_value = sum(x for i,x in feature_dict.iteritems() if col_name in i)
            
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i]
            for i in keys_to_remove:
                feature_dict.pop(i)
            feature_dict[col_name] = sum_value
    results = pd.Series(feature_dict.values(), index=feature_dict.keys())
    results.sort(axis=1)
    print results
    results.plot(kind='barh',figsize=(width,len(results)/4),xlim=(0,x_scale))

