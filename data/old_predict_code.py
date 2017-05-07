def predict(collection, classifier, name, ngrams):
    folds = 10
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for fold in range(0, folds):
        # Split data into training and test sets
        comment_train, comment_test, class_train, class_test = cross_validation(collection)

        # Vectorise the features of the training set by extract term frequencies from the comments
        if ngrams:
            count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        else:
            count_vectorizer = CountVectorizer()

        comment_train = collection.extract_features_cross(comment_train, count_vectorizer)
        classifier.fit(comment_train, class_train)

        # Vectorise the features of the test set by transforming the set
        comment_test = count_vectorizer.transform(comment_test)
        # Predict "spam" or "not spam" using the test set
        predictions = classifier.predict(comment_test)

        analysis = analyse_results(class_test, predictions, fold, name, ngrams)
        f1_scores.append(analysis[0])
        precision_scores.append(analysis[1])
        recall_scores.append(analysis[2])

    f1_result = (sum(f1_scores) / len(f1_scores)) * 100
    precision_result = (sum(precision_scores) / len(precision_scores)) * 100
    recall_result = (sum(recall_scores) / len(recall_scores)) * 100

    if ngrams:
        print "Results for --", name, "-- classifier over ", folds, " Folds (1-gram and 2-gram):"
    else:
        print "Results for --", name, "-- classifier over ", folds, " Folds:"
    print "F1 Score: ", f1_result, "%"
    print "Precision: ", precision_result, "%"
    print "Recall: ", recall_result, "%", "\n"

