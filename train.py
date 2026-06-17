import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#import local project components
#functions to load data and to visualize them
from utils.data_loader import load_bloodmnist_data, make_datasets
from utils.visualization import plot_learning_curves
#and model's architectures
from models.baseline_cnn import create_baseline_model
from models.augmented_cnn import create_augmented_model
from models.bn_cnn import create_bn_model
from models.attention_cnn import create_attention_model
from models.bn_noaug_cnn import create_bn_noaug_model
from models.attention_noaug_cnn import create_attention_noaug_model
from models.resnet50_transfer import create_resnet50_transfer_model

#set random seeds for numpy, tensorflow and python hashing to make results reproducible
def set_reproducibility(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_model_creator(model_key):
    creators = {
        'A': create_baseline_model,
        'B': create_augmented_model,
        'C': create_bn_model,
        'D': create_attention_model,
        'E': create_bn_noaug_model,
        'F': create_attention_noaug_model,
        'G': create_resnet50_transfer_model,
    }
    return creators.get(model_key.upper())


'''
A: cnn 2 layers
B: cnn 2 lyers + data augmentation
C: cnn 2 layers + data augmentation + batch normalization
D: cnn 2 layers + data augmentation + batch normalization + spatial attention
E: cnn 2 layers + batch normalization
F: cnn 2 layers + batch normalization + spatial attention
G: resnet50 fine tuned
'''


def main():
    #this first part allows to run the script from the terminal:
    #e.g. python main.py --model all --epochs 20 --batch-size 64
    parser = argparse.ArgumentParser(description="Train BloodMNIST CNN Pipeline Models")
    parser.add_argument('--model', type=str, default='all', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'all'],
                        help="Model letter to train: A, B, C, D, E, F, G or all (default: all)")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Maximum training epochs (default: 50)")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    set_reproducibility(args.seed)

    #create output directories
    os.makedirs("saved_models", exist_ok=True) #here the trained models will be saved
    os.makedirs("results/curves", exist_ok=True) #here the training plots will be saved

    #load data
    img_size = 64 #this can also be set to different size supported by the midmnist dataset
    x_train, y_train, x_val, y_val, x_test, y_test = load_bloodmnist_data(img_size=img_size)

    #calls the function from utils/data_loader.py which loads the data, splits it into train/validation/test
    #normalizes the images and converts them into tensorflow datasets, shuffled (only training set), batched and prefetched
    train_ds, val_ds, test_ds = make_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=args.batch_size, seed=args.seed
    )

    #models to train depending on users inputs from terminal
    models_to_train = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] if args.model.lower() == 'all' else [args.model.upper()]

    #loops the training over the selected models
    for key in models_to_train:
        print(f"\n=======================================================")
        print(f" TRAINING MODEL {key}")
        print(f"=======================================================")
        
        set_reproducibility(args.seed)
        
        #create the models using the scripts contained in models/
        model_creator = get_model_creator(key)
        model = model_creator(input_shape=(img_size, img_size, 3), num_classes=8)
        model.summary()

        #parameter counts
        #TensorFlows stores all trainable tensors inside: model.trainable_variables
        trainable_params = int(sum([tf.size(variable).numpy() for variable in model.trainable_variables]))
        #and non trinable parameters inside model.non_trainable_parameters
        non_trainable_params = int(sum([tf.size(variable).numpy() for variable in model.non_trainable_variables]))
        total_params = trainable_params + non_trainable_params
        
        print(f"Total Params: {total_params:,}")
        print(f"Trainable Params: {trainable_params:,}")
        print(f"Non-Trainable Params: {non_trainable_params:,}")

        #compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy', #because labels are integer class IDs, not one-hot encoded vectors
            metrics=['accuracy']
        )

        #callbacks
        
        #if validation loss does not improve for 8 epochs,
        #training stops early and the model goes back to the weights
        #from the best validation-loss epoch
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        )
        #if validation loss does not improve for 4 epochs,
        #the learning rate is multiplied by 0.2,
        #but it will not go below 1e-6
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=4, 
            min_lr=1e-6,
            verbose=1
        )
        callbacks = [early_stopping, lr_scheduler]

        #train
        start_time = time.time() #save starting time
        history = model.fit( #history stores loss, accuracy, validation loss and validation accuracy
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )
        end_time = time.time() #save ending time

        total_time = end_time - start_time
        num_epochs_trained = len(history.history['loss'])
        avg_epoch_time = total_time / num_epochs_trained if num_epochs_trained > 0 else 0

        print(f"\nTraining Complete for Model {key}!")
        print(f"Total training time: {total_time:.2f} seconds ({total_time/60.0:.2f} minutes)")
        print(f"Average epoch time:  {avg_epoch_time:.2f} seconds")

        #save models' weights and structure
        model_filename = f"model_{key.lower()}.keras"
        model_path = os.path.join("saved_models", model_filename)
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        #model file size
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"Model file size: {file_size_mb:.2f} MB")

        #creates and saves plots using functions from utils/visualization
        plot_learning_curves(history.history, model.name, "results/curves")

        #save training metadata to .JSON file, which will be stored in results/ directory
        metadata = {
            'model_letter': key,
            'model_name': model.name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'file_size_mb': float(f"{file_size_mb:.2f}"),
            'total_training_time_sec': float(f"{total_time:.2f}"),
            'avg_epoch_time_sec': float(f"{avg_epoch_time:.2f}"),
            'epochs_trained': num_epochs_trained,
            'history': {metric: [float(val) for val in values] for metric, values in history.history.items()}
        }

        metadata_path = os.path.join("results", f"metadata_{key.lower()}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}\n")

if __name__ == '__main__':
    main()
