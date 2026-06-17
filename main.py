import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

#import local project components
#functions to load data and to visualize them
from utils.data_loader import load_bloodmnist_data
from utils.visualization import plot_confusion_matrix

#BloodMNIST class names
CLASS_NAMES = [
    'Basophil', 
    'Eosinophil', 
    'Erythroblast', 
    'Immature Granulocyte', 
    'Lymphocyte', 
    'Monocyte', 
    'Neutrophil', 
    'Platelet'
]

def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Trained BloodMNIST Models")
    args = parser.parse_args()

    print("\n=======================================================")
    print(" EVALUATION & COMPARISON ENGINE")
    print("=======================================================")

    #load test data using load_blood from utils/data_loader
    img_size = 64
    _, _, _, _, x_test, y_test = load_bloodmnist_data(img_size=img_size)


    #create results/ directory to store plots and output metrics
    os.makedirs("results/evaluation", exist_ok=True)
    #choose models o evaluate
    models_to_evaluate = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    #initialize a dictionary that will store the final metrics for all evaluated models
    comparison_results = {}
    #loops over the models to be evaluated
    for key in models_to_evaluate:
        model_filename = f"model_{key}.keras"
        model_path = os.path.join("saved_models", model_filename)
        metadata_path = os.path.join("results", f"metadata_{key}.json")

        if not os.path.exists(model_path):
            print(f"Warning: Trained model file {model_path} not found. Skipping.")
            continue

        print(f"\nEvaluating Model {key.upper()}...")
        
        #load model [custom object and safe_mode where necessary to allow to load the custom spatial attention block of model D]
        model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf}, safe_mode=False)
        
        #load the already stored training metadata
        training_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                training_metadata = json.load(f)
        else:
            print(f"Warning: Training metadata {metadata_path} not found. Some training-time metrics will be missing.")

        #warm-up
        print("Performing GPU/CPU warm-up prediction...")
        _ = model.predict(x_test[:10], verbose=0)

        #measure total prediction time on the entire test set
        print("Measuring inference latency...")
        start_latency = time.perf_counter()
        y_pred_probs = model.predict(x_test, batch_size=64, verbose=0)
        end_latency = time.perf_counter()
        
        total_inf_time = end_latency - start_latency
        avg_inf_time_per_image_ms = (total_inf_time / len(x_test)) * 1000.0

        #also measure single unbatched image inference latency 
        single_img_times = []
        num_single_tests = min(100, len(x_test))
        for i in range(num_single_tests):
            single_img = np.expand_dims(x_test[i], axis=0)
            start_s = time.perf_counter()
            _ = model.predict(single_img, verbose=0)
            end_s = time.perf_counter()
            single_img_times.append((end_s - start_s) * 1000.0)
        avg_single_inf_time_ms = np.mean(single_img_times)

        #get hard predictions
        y_pred = np.argmax(y_pred_probs, axis=1)

        #performance metrics
        acc = accuracy_score(y_test, y_pred)
        
        #compute multi-class AUC (One-vs-Rest macro average)
        try:
            auc_score = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"AUC calculation failed (possibly due to dry-run class slice limits): {e}")
            auc_score = 0.5

        #Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        print(f"Results for Model {key.upper()}:")
        print(f"  Test Accuracy: {acc*100:.2f}%")
        print(f"  Test AUC (Macro): {auc_score:.4f}")
        print(f"  F1-Score (Macro): {f1:.4f}")
        print(f"  Inference Latency (Batched):  {avg_inf_time_per_image_ms:.3f} ms / image")
        print(f"  Inference Latency (Unbatched): {avg_single_inf_time_ms:.3f} ms / image")

        #plot and save confusion matrices in results/ directory
        model_name = training_metadata.get('model_name', model.name)
        plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, model_name, "results/evaluation")

        #populate the comparison dictionary
        comparison_results[key.upper()] = {
            'model_name': model_name,
            'total_params': training_metadata.get('total_params', model.count_params()),
            'trainable_params': training_metadata.get('trainable_params', 'N/A'),
            'file_size_mb': training_metadata.get('file_size_mb', 'N/A'),
            'total_train_time_sec': training_metadata.get('total_training_time_sec', 0.0),
            'epochs_trained': training_metadata.get('epochs_trained', 0),
            'test_accuracy': acc,
            'test_auc': auc_score,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'latency_batched_ms': avg_inf_time_per_image_ms,
            'latency_unbatched_ms': avg_single_inf_time_ms
        }

    #exit if no models were already trained. The train.py file needs to be ran before this one
    if not comparison_results:
        print("\nNo models were evaluated. Please run train.py first.")
        return

    #save comparison in a  .JSON file, that will be stored in results/ directory
    with open("results/comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=4)



if __name__ == '__main__':
    main()
