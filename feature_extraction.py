from sklearn.feature_extraction import DictVectorizer

def extract_features(data):
    # Convert the dataframe to a list of dictionaries for feature extraction
    data_dict = data.to_dict('records')

    # Use DictVectorizer to extract features
    vec = DictVectorizer(sparse=False)
    data_features = vec.fit_transform(data_dict)

    # Convert back to DataFrame for easier handling
    feature_names = vec.get_feature_names_out()
    data_features_df = pd.DataFrame(data_features, columns=feature_names)

    return data_features_df, feature_names
