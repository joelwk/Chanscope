import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def self_train(train_data, train_labels, val_data, val_labels, unlabeled_data, params, spam_clusters, classifier, save_dir='saved_models'):
    # Define a function to filter uncertainties
    def filter_uncertainties(data):
        # Select for the unlabeled data points that are closest to the centroids of the spam clusters
        uncertanty_data = data[data['cluster'].isin(spam_clusters) & data['spam_label'].isin([''])]
        unlabeled_samples, max_length = classifier.load_and_process_data(data, 512)
        unlabeled_data = np.stack(unlabeled_samples['df']['vector'])
        unlabeled_data = tf.data.Dataset.from_tensor_slices(unlabeled_data)
        unlabeled_data = unlabeled_data.batch(10) 
        return unlabeled_data, max_length

    # Define a function to reset the weights of the model
    def reset_weights(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                reset_weights(layer)
                continue
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                # find the corresponding variable
                var = getattr(layer, k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
                
    # Define a new model that includes dropout during inference
    class DropoutInference(tf.keras.Model):
        def __init__(self, teacher_model):
            super(DropoutInference, self).__init__()
            self.teacher_model = teacher_model

        def call(self, inputs, training=False):
            return self.teacher_model(inputs, training=True)

    unlabeled_data, max_length = filter_uncertainties(unlabeled_data)

    num_iterations = 5 
    evaluation_metrics = []  # list to store evaluation metrics for each iteration
    save_dir = 'saved_models'

    # Check if student_model is already defined
    if 'student_model' not in locals():
        student_model = None
    histories = [] 
    for iteration in range(num_iterations):
        if student_model is None and iteration == 0:
            student_model = tf.keras.models.load_model(os.path.join(save_dir, 'best_model.tf'))
        elif iteration > 0:
            reset_weights(student_model)
        # Step 2: Shuffle the training data (outside the if condition)
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        # Step 3: Train the student model (calling the function directly)
        student_history, student_model = classifier.train_model(
            train_data, 
            train_labels, 
            val_data, 
            val_labels, 
            params
            )
        histories.append(student_history)
        student_model.save(os.path.join(save_dir, f"student_model_{iteration}.tf"))
        # Evaluate the student model on validation data
        evaluation = student_model.evaluate(val_data, val_labels)
        evaluation_metrics.append(evaluation)  # save evaluation metrics
            
        # Step 3: Generate pseudo-labels
        dropout_model = DropoutInference(student_model)
        all_preds = []
        for _ in range(params['T']):
            preds = []
            # Convert unlabeled_data to list of numpy arrays for indexing
            unlabeled_data_list = [sample.numpy() for batch in unlabeled_data for sample in batch]
            # Flatten each sample and then reshape it to have a sequence length of 32 and a feature size of 512
            unlabeled_data_list_padded = [sample.flatten() for sample in unlabeled_data_list]
            # Stack the list of padded samples into a single numpy array
            unlabeled_data_array = np.stack(unlabeled_data_list_padded)
                
            batch_predictions = dropout_model.predict(unlabeled_data_array)
            predicted_labels = np.argmax(batch_predictions, axis=1)
            preds.extend(predicted_labels)
            all_preds.append(preds)

        # Step 4: Evaluate uncertainty and update labels

        # Compute variances and find the most uncertain sampls
        variances = np.zeros(len(unlabeled_data))
        # Calculate the variance for each data point
        for i in range(len(unlabeled_data)):
            variances[i] = np.var([preds[i] for preds in all_preds if preds[i] is not None])
        # Get the index of the sample with the highest variance
        # Define a threshold for uncertainty
        uncertainty_threshold = 0.05

        # Select samples that are below the uncertainty threshold
        selected_indices = np.where(variances < uncertainty_threshold)[0]

        # Select samples and labels
        selected_samples = unlabeled_data_array[selected_indices]
        selected_labels = [np.bincount([preds[i] for preds in all_preds if preds[i] is not None]).argmax() for i in selected_indices]

        # Remove the selected samples from unlabeled_data_array
        unlabeled_data_array = np.delete(unlabeled_data_array, selected_indices, axis=0)

        # Update train_data and train_labels
        train_data = np.concatenate((train_data, selected_samples))
        train_labels = np.concatenate((train_labels, selected_labels))

        # The student model becomes the teacher model for the next iteration
        teacher_model = student_model

        return teacher_model, evaluation_metrics, histories
